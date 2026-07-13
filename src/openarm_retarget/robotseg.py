from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .media import _iter_bgr_frames, _video_info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segment_robotseg_video(
    video_path: str | Path,
    output_dir: str | Path,
    repository: str | Path,
    checkpoint: str | Path,
    *,
    category: str = "robot",
    chunk_frames: int = 120,
    device: int = 0,
) -> Path:
    """Run the official RobotSeg video model in bounded-memory independent chunks.

    RobotSeg is intentionally kept as a pinned external checkout. Independent chunks bound
    memory and re-run its learned automatic robot prompt, limiting long-video drift. Its output
    is particularly useful as a high-precision robot-only contact mask; generic SAM masks may be
    unioned separately when maximum removal recall is required.
    """
    if category not in {"robot", "arm", "gripper"}:
        raise ValueError("category must be robot, arm, or gripper")
    if chunk_frames < 2:
        raise ValueError("chunk_frames must be at least two")
    repository = Path(repository).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not (repository / "robotseg/build_robotseg.py").is_file():
        raise FileNotFoundError(f"Not a RobotSeg checkout: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        import torch
        import hydra  # noqa: F401
        import iopath  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Install RobotSeg support with: uv sync --extra media-ai --extra robotseg"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("RobotSeg requires a CUDA GPU in this pipeline")
    sys.path.insert(0, str(repository))
    try:
        from robotseg.build_robotseg import build_robotseg_video_predictor
    finally:
        sys.path.pop(0)

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _video_info(video_path)
    width, height = int(info["width"]), int(info["height"])
    torch.cuda.set_device(device)
    predictor = build_robotseg_video_predictor(
        "configs/robotseg-infer", str(checkpoint), device=f"cuda:{device}"
    )
    frame_iterator = iter(_iter_bgr_frames(video_path, width, height))
    global_index = 0
    chunks = []
    while True:
        frames = []
        for _ in range(chunk_frames):
            frame = next(frame_iterator, None)
            if frame is None:
                break
            frames.append(frame)
        if not frames:
            break
        with tempfile.TemporaryDirectory(prefix="openarm-robotseg-") as temporary:
            frame_dir = Path(temporary)
            for local_index, frame in enumerate(frames):
                if not cv2.imwrite(
                    str(frame_dir / f"{local_index:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError("Could not write a RobotSeg input frame")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(
                    video_path=str(frame_dir),
                    async_loading_frames=False,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                )
                predictor.add_new_robot(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=0,
                    robot=category,
                )
                written = 0
                for local_index, _object_ids, logits in predictor.propagate_in_video(
                    inference_state=state, robot=category
                ):
                    mask = (logits[0] > 0).squeeze().cpu().numpy().astype(np.uint8) * 255
                    if not cv2.imwrite(
                        str(output_dir / f"{global_index + int(local_index):06d}.png"), mask
                    ):
                        raise RuntimeError("Could not write a RobotSeg output mask")
                    written += 1
            if written != len(frames):
                raise RuntimeError(
                    f"RobotSeg returned {written} masks for {len(frames)} input frames"
                )
        chunks.append({"start_frame": global_index, "frames": len(frames), "category": category})
        global_index += len(frames)
        torch.cuda.empty_cache()
    if global_index == 0:
        raise RuntimeError("Video decoder returned zero frames")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    manifest = {
        "source_video": str(video_path),
        "output_format": "six-digit PNG, 0 background, 255 robot",
        "frames": global_index,
        "fps": float(info["fps"]),
        "resolution": [width, height],
        "category": category,
        "chunk_frames": chunk_frames,
        "device": device,
        "repository": "https://github.com/showlab/RobotSeg",
        "repository_revision": revision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "chunks": chunks,
    }
    manifest_path = output_dir / "robotseg_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
