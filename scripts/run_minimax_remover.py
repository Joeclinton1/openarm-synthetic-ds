#!/usr/bin/env python3
"""Fast windowed MiniMax-Remover inference with hard background preservation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "MiniMax-Remover"
sys.path.insert(0, str(VENDOR))

from pipeline_minimax_remover import Minimax_Remover_Pipeline  # noqa: E402
from transformer_minimax_remover import Transformer3DModel  # noqa: E402


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames), fps


def read_masks(path: Path, count: int, shape: tuple[int, int]) -> np.ndarray:
    masks = []
    for index in range(count):
        mask = cv2.imread(str(path / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            raise FileNotFoundError(path / f"{index:06d}.png")
        masks.append(mask > 127)
    return np.stack(masks)


def window_starts(frame_count: int, length: int, stride: int) -> list[int]:
    if frame_count <= length:
        return [0]
    starts = list(range(0, frame_count - length + 1, stride))
    final = frame_count - length
    if starts[-1] != final:
        starts.append(final)
    return starts


def load_pipeline(
    model: str, *, gpu_id: int, local_files_only: bool
) -> Minimax_Remover_Pipeline:
    device = torch.device(f"cuda:{gpu_id}")
    vae = AutoencoderKLWan.from_pretrained(
        model,
        subfolder="vae",
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    )
    transformer = Transformer3DModel.from_pretrained(
        model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        model, subfolder="scheduler", local_files_only=local_files_only
    )
    return Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler).to(
        device
    )


def run_clip(
    pipe: Minimax_Remover_Pipeline,
    video_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    steps: int,
    dilation: int,
    window: int,
    stride: int,
    seed: int,
    gpu_id: int,
    max_windows: int | None,
) -> dict:
    started = time.perf_counter()
    frames, fps = read_video(video_path)
    masks = read_masks(mask_path, len(frames), frames.shape[1:3])
    height, width = frames.shape[1:3]
    starts = window_starts(len(frames), window, stride)
    if max_windows is not None:
        starts = starts[:max_windows]
    accumulated = np.zeros_like(frames, dtype=np.float32)
    weights = np.zeros((len(frames), 1, 1, 1), dtype=np.float32)
    timings = []
    for start in starts:
        stop = min(start + window, len(frames))
        count = stop - start
        images = frames[start:stop]
        window_masks = masks[start:stop]
        if count < window:
            padding = window - count
            images = np.concatenate([images, np.repeat(images[-1:], padding, axis=0)])
            window_masks = np.concatenate(
                [window_masks, np.repeat(window_masks[-1:], padding, axis=0)]
            )
        before = time.perf_counter()
        generated = pipe(
            images=torch.from_numpy(images.astype(np.float32) / 127.5 - 1.0),
            masks=torch.from_numpy(window_masks[..., None].astype(np.float32)),
            num_frames=window,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed + start),
            iterations=dilation,
        ).frames[0]
        generated = np.asarray(generated[:count])
        if generated.dtype != np.uint8:
            generated = np.clip(generated * 255.0, 0, 255).astype(np.uint8)
        # Overlapping windows use a triangular confidence profile.
        profile = np.minimum(np.arange(count) + 1, np.arange(count, 0, -1)).astype(np.float32)
        profile /= max(float(profile.max()), 1.0)
        accumulated[start:stop] += generated.astype(np.float32) * profile[:, None, None, None]
        weights[start:stop] += profile[:, None, None, None]
        timings.append(time.perf_counter() - before)
    missing = weights[:, 0, 0, 0] == 0
    generated_all = np.divide(
        accumulated,
        np.maximum(weights, 1e-6),
        out=np.zeros_like(accumulated),
    )
    generated_all[missing] = frames[missing]
    # The generative model owns only the dilated robot region; source pixels are exact elsewhere.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    constrained = frames.copy()
    for index, mask in enumerate(masks):
        safe = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        constrained[index][safe] = np.clip(generated_all[index][safe], 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")
    for frame in constrained:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    manifest = {
        "method": "MiniMax-Remover with overlap weighting and hard background constraint",
        "source": str(video_path.resolve()),
        "masks": str(mask_path.resolve()),
        "frames": len(frames),
        "fps": fps,
        "resolution": [width, height],
        "window_frames": window,
        "window_stride": stride,
        "window_starts": starts,
        "inference_steps": steps,
        "mask_dilation_iterations": dilation,
        "gpu_id": gpu_id,
        "window_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.with_suffix(".minimax.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("masks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="zibojia/minimax-remover")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--dilation", type=int, default=4)
    parser.add_argument("--window", type=int, default=81)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads instead of requiring locally cached model weights",
    )
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()
    if args.window % 4 != 1:
        raise ValueError("Window length must be 4k+1 for the Wan temporal VAE")
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    pipeline = load_pipeline(
        args.model,
        gpu_id=args.gpu_id,
        local_files_only=not args.allow_download,
    )
    result = run_clip(
        pipeline,
        args.video,
        args.masks,
        args.output,
        steps=args.steps,
        dilation=args.dilation,
        window=args.window,
        stride=args.stride,
        seed=args.seed,
        gpu_id=args.gpu_id,
        max_windows=args.max_windows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
