#!/usr/bin/env python3
"""Composite aligned OpenArm renders and rebuild the labelled three-panel review videos."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from openarm_retarget.media import composite_video


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "outputs/cross_dataset_openarm_benchmark"


def _right_edge_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8
    )
    width = mask.shape[1]
    candidates = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH])
        >= width - 2
    ]
    if not candidates:
        return np.zeros(mask.shape, dtype=np.float32)
    label = max(candidates, key=lambda item: int(stats[item, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.float32)
    return cv2.GaussianBlur(component, (0, 0), sigmaX=1.0)


def _restore_stationary_agibot_arm(clip: Path) -> Path:
    """Restore only the stationary right source arm over the fully removed review plate."""
    source_capture = cv2.VideoCapture(str(clip / "01_source.mp4"))
    removed_capture = cv2.VideoCapture(str(clip / "02_robot_removed.mp4"))
    fps = float(source_capture.get(cv2.CAP_PROP_FPS))
    width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = clip / "02_active_arm_removed.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frames = 0
    try:
        while True:
            source_ok, source = source_capture.read()
            removed_ok, removed = removed_capture.read()
            if not source_ok and not removed_ok:
                break
            if not source_ok or not removed_ok:
                raise RuntimeError("AgiBot source and removed videos have different lengths")
            mask = cv2.imread(
                str(clip / "masks_final" / f"{frames:06d}.png"), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                raise FileNotFoundError(clip / "masks_final" / f"{frames:06d}.png")
            stationary = _right_edge_component(mask)[..., None]
            restored = np.clip(
                source.astype(np.float32) * stationary
                + removed.astype(np.float32) * (1.0 - stationary),
                0,
                255,
            ).astype(np.uint8)
            writer.write(restored)
            frames += 1
    finally:
        source_capture.release()
        removed_capture.release()
        writer.release()
    if frames == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("AgiBot video decoder returned zero frames")
    return output


def compose_clip(clip: Path) -> None:
    source = clip / "01_source.mp4"
    removed = clip / "02_robot_removed.mp4"
    aligned = clip / "render_aligned"
    output = clip / "03_openarm_output.mp4"
    triplet = clip / "source_removed_openarm.mp4"
    background = (
        _restore_stationary_agibot_arm(clip)
        if clip.parent.name == "agibot_world_alpha"
        else removed
    )
    composite_video(background, aligned, output, linear_light=True)
    filters = []
    labels = ("SOURCE", "ROBOT REMOVED", "OPENARM OUTPUT")
    for index, label in enumerate(labels):
        filters.append(
            f"[{index}:v]drawtext=text='{label}':x=8:y=8:fontsize=16:"
            "fontcolor=white:borderw=2:bordercolor=black[v" + str(index) + "]"
        )
    filters.append("[v0][v1][v2]hstack=inputs=3[review]")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-i",
            str(removed),
            "-i",
            str(output),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[review]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(triplet),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    args = parser.parse_args()
    for clip in sorted(args.benchmark.resolve().glob("*/*")):
        if not (clip / "render_aligned").is_dir():
            continue
        compose_clip(clip)
        print(clip.relative_to(args.benchmark.resolve()))


if __name__ == "__main__":
    main()
