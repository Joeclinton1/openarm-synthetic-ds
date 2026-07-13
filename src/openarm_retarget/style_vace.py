from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
DEFAULT_PROMPT = (
    "Photorealistic OpenArm 2.0 dual robot arms performing tabletop manipulation, "
    "preserve the exact arm pose, silhouette, camera, object contact, scene, and lighting; "
    "dark graphite links, metallic joints, realistic material response and contact shadows"
)
DEFAULT_NEGATIVE_PROMPT = (
    "changed pose, changed silhouette, extra robot, extra arm, extra gripper, missing link, "
    "deformed geometry, floating robot, changed object, changed background, text, watermark, "
    "flicker, blur, low quality, cartoon"
)


def _probe(path: Path) -> tuple[int, int, float]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(Fraction(stream["avg_frame_rate"]))


def _frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def plan_style_chunks(total_frames: int, chunk_frames: int = 49, overlap: int = 8) -> list[dict]:
    """Plan fixed-size overlapping VACE windows and non-overlapping retained centers."""
    if total_frames < chunk_frames:
        raise ValueError("Video is shorter than one style window")
    if chunk_frames < 5 or (chunk_frames - 1) % 4:
        raise ValueError("chunk_frames must equal 4*k+1")
    if overlap < 1 or overlap >= chunk_frames:
        raise ValueError("overlap must be in [1, chunk_frames)")
    last_start = total_frames - chunk_frames
    starts = list(range(0, last_start + 1, chunk_frames - overlap))
    if starts[-1] != last_start:
        starts.append(last_start)
    jobs = [
        {"index": index, "start": start, "end": start + chunk_frames}
        for index, start in enumerate(starts)
    ]
    for index, job in enumerate(jobs):
        previous_end = jobs[index - 1]["end"] if index else 0
        next_start = jobs[index + 1]["start"] if index + 1 < len(jobs) else total_frames
        job["keep_start"] = job["start"] if index == 0 else (previous_end + job["start"]) // 2
        job["keep_end"] = job["end"] if index + 1 == len(jobs) else (job["end"] + next_start) // 2
    assert jobs[0]["keep_start"] == 0 and jobs[-1]["keep_end"] == total_frames
    assert all(first["keep_end"] == second["keep_start"] for first, second in zip(jobs, jobs[1:]))
    return jobs


def _run_batch_worker(arguments: dict) -> list[str]:
    results = []
    for job in arguments.pop("jobs"):
        output = Path(arguments["output_dir"]) / f"chunk_{job['index']:04d}_{job['start']:06d}.mp4"
        run_vace_style_clip(
            arguments["input_video"],
            arguments["rgba_dir"],
            output,
            protected_mask_dir=arguments["protected_mask_dir"],
            start_frame=job["start"],
            num_frames=job["end"] - job["start"],
            dilation_px=arguments["dilation_px"],
            model_id=arguments["model_id"],
            prompt=arguments["prompt"],
            negative_prompt=arguments["negative_prompt"],
            steps=arguments["steps"],
            guidance_scale=arguments["guidance_scale"],
            conditioning_scale=arguments["conditioning_scale"],
            # Keep one clip-level noise identity across overlapping windows. The retained-center
            # stitch plus the temporal gate handle boundary context without intentional style drift.
            seed=arguments["seed"],
            gpu_id=arguments["gpu_id"],
        )
        results.append(str(output))
    return results


def _read_clip(path: Path, start_frame: int, num_frames: int) -> tuple[list[Image.Image], float]:
    width, height, fps = _probe(path)
    select = f"select=between(n\\,{start_frame}\\,{start_frame + num_frames - 1})"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            select,
            "-frames:v",
            str(num_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Could not read FFmpeg output")
    frame_bytes = width * height * 3
    frames: list[Image.Image] = []
    while len(frames) < num_frames:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            process.kill()
            raise RuntimeError("FFmpeg returned a partial frame")
        frames.append(Image.fromarray(np.frombuffer(raw, np.uint8).reshape(height, width, 3)))
    stderr = process.communicate()[1].decode(errors="replace")
    if process.returncode:
        raise RuntimeError(f"FFmpeg failed while reading {path}: {stderr}")
    if len(frames) != num_frames:
        raise ValueError(f"Requested {num_frames} frames at {start_frame}, received {len(frames)}")
    return frames, fps


def _load_masks(
    rgba_dir: Path,
    start_frame: int,
    num_frames: int,
    size: tuple[int, int],
    dilation_px: int,
    protected_mask_dir: Path | None,
) -> list[Image.Image]:
    width, height = size
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    masks: list[Image.Image] = []
    for index in range(start_frame, start_frame + num_frames):
        rgba = cv2.imread(str(rgba_dir / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape != (height, width, 4):
            raise ValueError(f"Missing or invalid RGBA control frame {index:06d}")
        mask = rgba[..., 3] > 7
        if dilation_px:
            mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        if protected_mask_dir is not None:
            protected = cv2.imread(
                str(protected_mask_dir / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE
            )
            if protected is None or protected.shape != (height, width):
                raise ValueError(f"Missing or invalid protected mask frame {index:06d}")
            mask &= protected == 0
        # VACE semantics: black pixels are preserved; white pixels are generated.
        masks.append(Image.fromarray(mask.astype(np.uint8) * 255, mode="L"))
    return masks


def _reference_images(rgba_dir: Path, indices: list[int], size: tuple[int, int]) -> list[Image.Image]:
    width, height = size
    images: list[Image.Image] = []
    for index in indices:
        rgba = cv2.imread(str(rgba_dir / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape != (height, width, 4):
            raise ValueError(f"Missing or invalid reference RGBA frame {index:06d}")
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        neutral = np.full((height, width, 3), 127, dtype=np.float32)
        bgr = rgba[..., :3].astype(np.float32) * alpha + neutral * (1 - alpha)
        images.append(Image.fromarray(cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)))
    return images


def _write_video(frames: list[np.ndarray], output: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No generated frames")
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("Could not open FFmpeg input")
    try:
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError("Generated frame resolutions differ")
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        process.stdin = None
        stderr = process.communicate()[1].decode(errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
    if process.returncode:
        raise RuntimeError(f"FFmpeg could not encode VACE output: {stderr}")


def run_vace_style_clip(
    input_video: str | Path,
    rgba_dir: str | Path,
    output: str | Path,
    *,
    protected_mask_dir: str | Path | None = None,
    start_frame: int = 0,
    num_frames: int = 49,
    dilation_px: int = 3,
    reference_indices: list[int] | None = None,
    model_id: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    steps: int = 20,
    guidance_scale: float = 4.0,
    conditioning_scale: float = 1.0,
    seed: int = 17,
    gpu_id: int = 0,
) -> Path:
    """Run a bounded VACE appearance experiment around an accepted deterministic render.

    This produces a candidate only. ``apply_mask_constrained_style`` must be used before a
    candidate can become release output; VACE is never allowed to become the pose authority.
    """
    if num_frames < 5 or (num_frames - 1) % 4:
        raise ValueError("Wan VACE requires num_frames = 4*k+1 (for example 17, 49, or 81)")
    if start_frame < 0 or steps < 1 or dilation_px < 0 or gpu_id < 0:
        raise ValueError("start_frame, steps, dilation_px, and gpu_id must be non-negative")
    input_path, rgba_root, output_path = Path(input_video), Path(rgba_dir), Path(output)
    frames, fps = _read_clip(input_path, start_frame, num_frames)
    width, height = frames[0].size
    if width % 16 or height % 16:
        raise ValueError("VACE input width and height must be divisible by 16")
    protected_root = Path(protected_mask_dir) if protected_mask_dir else None
    masks = _load_masks(
        rgba_root,
        start_frame,
        num_frames,
        (width, height),
        dilation_px,
        protected_root,
    )
    if reference_indices is None:
        reference_indices = [start_frame + num_frames // 2]
    references = _reference_images(rgba_root, reference_indices, (width, height))

    try:
        import diffusers
        import torch
        import transformers
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("Install the style model dependencies with: uv sync --extra style-ai") \
            from error

    started = time.monotonic()
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    pipe.vae.enable_tiling()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        video=frames,
        mask=masks,
        reference_images=references,
        conditioning_scale=conditioning_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
    ).frames[0]
    generated = [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in result]
    _write_video(generated, output_path, fps)
    manifest = {
        "schema": "openarm-vace-style-candidate-v1",
        "release_accepted": False,
        "geometry_authority": str(rgba_root.resolve()),
        "input_video": str(input_path.resolve()),
        "output_video": str(output_path.resolve()),
        "protected_masks": str(protected_root.resolve()) if protected_root else None,
        "model_id": model_id,
        "start_frame": start_frame,
        "num_frames": num_frames,
        "reference_indices": reference_indices,
        "dilation_px": dilation_px,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "conditioning_scale": conditioning_scale,
        "seed": seed,
        "gpu_id": gpu_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "runtime_seconds": time.monotonic() - started,
        "software": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        },
        "policy": "candidate only; mask-constrain, independently validate, or reject",
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return output_path


def run_vace_style_batch(
    input_video: str | Path,
    rgba_dir: str | Path,
    output_dir: str | Path,
    *,
    protected_mask_dir: str | Path | None = None,
    chunk_frames: int = 81,
    overlap: int = 8,
    gpu_ids: tuple[int, ...] = (0, 1),
    dilation_px: int = 3,
    model_id: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    steps: int = 8,
    guidance_scale: float = 3.5,
    conditioning_scale: float = 1.0,
    seed: int = 17,
) -> Path:
    """Generate all raw VACE windows, with one serial worker process per GPU."""
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("gpu_ids must contain unique non-negative device indices")
    input_path, rgba_root = Path(input_video).resolve(), Path(rgba_dir).resolve()
    protected_root = Path(protected_mask_dir).resolve() if protected_mask_dir else None
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    total_frames = _frame_count(input_path)
    jobs = plan_style_chunks(total_frames, chunk_frames, overlap)
    worker_arguments = []
    for worker_index, gpu_id in enumerate(gpu_ids):
        worker_arguments.append(
            {
                "jobs": jobs[worker_index :: len(gpu_ids)],
                "gpu_id": gpu_id,
                "input_video": str(input_path),
                "rgba_dir": str(rgba_root),
                "output_dir": str(destination),
                "protected_mask_dir": str(protected_root) if protected_root else None,
                "dilation_px": dilation_px,
                "model_id": model_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "conditioning_scale": conditioning_scale,
                "seed": seed,
            }
        )
    started = time.monotonic()
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=context) as pool:
        outputs = [path for paths in pool.map(_run_batch_worker, worker_arguments) for path in paths]
    for job in jobs:
        job["candidate"] = str(
            destination / f"chunk_{job['index']:04d}_{job['start']:06d}.mp4"
        )
    manifest = {
        "schema": "openarm-vace-style-batch-v1",
        "release_accepted": False,
        "input_video": str(input_path),
        "rgba_geometry_authority": str(rgba_root),
        "protected_masks": str(protected_root) if protected_root else None,
        "total_frames": total_frames,
        "chunk_frames": chunk_frames,
        "overlap": overlap,
        "gpu_ids": list(gpu_ids),
        "steps": steps,
        "guidance_scale": guidance_scale,
        "conditioning_scale": conditioning_scale,
        "dilation_px": dilation_px,
        "model_id": model_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "jobs": jobs,
        "outputs": outputs,
        "runtime_seconds": time.monotonic() - started,
        "policy": "raw candidates only; constrain and independently validate merged output",
    }
    manifest_path = destination / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
