#!/usr/bin/env python3
"""Compute auditable preservation, temporal, kinematic, and runtime metrics."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "outputs/cross_dataset_openarm_benchmark"


def read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def removal_runtime(clip: Path) -> tuple[str, float]:
    static = clip / "02_robot_removed.static.json"
    minimax = clip / "02_robot_removed.minimax.json"
    if static.exists():
        payload = json.loads(static.read_text())
        return "static_clean_plate", float(payload["runtime_seconds"])
    payload = json.loads(minimax.read_text())
    return "minimax_remover", float(payload["elapsed_seconds"])


def render_runtime(clip: Path) -> float:
    side_paths = [clip / f"render_raw_{side}/render_manifest.json" for side in ("left", "right")]
    if all(path.exists() for path in side_paths):
        return max(float(json.loads(path.read_text())["seconds"]) for path in side_paths)
    return float(json.loads((clip / "render_raw/render_manifest.json").read_text())["seconds"])


def kinematic_metrics(trajectory: Path) -> dict[str, float]:
    with np.load(trajectory, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        active_names = metadata.get("active_sides", ["right", "left"])
        active = [0 if side == "right" else 1 for side in active_names]
        feasible = archive["feasible"].astype(bool)
        position = archive["diagnostic_position_error_m"][:, active]
        orientation = archive["diagnostic_orientation_error_rad"][:, active]
    return {
        "feasible_fraction": float(feasible.mean()),
        "position_error_p95_mm": float(np.quantile(position, 0.95) * 1000.0),
        "orientation_error_p95_rad": float(np.quantile(orientation, 0.95)),
    }


def evaluate_clip(clip: Path) -> dict[str, float | int | str | bool]:
    source = read_frames(clip / "01_source.mp4")
    removed = read_frames(clip / "02_robot_removed.mp4")
    output = read_frames(clip / "03_openarm_output.mp4")
    if not len(source) == len(removed) == len(output):
        raise ValueError(f"Frame mismatch in {clip}")
    background_abs_sum = background_squared_sum = 0.0
    background_values = 0
    masked_abs_sum = 0.0
    masked_values = 0
    output_background_abs_sum = 0.0
    output_background_values = 0
    temporal_abs_sum = 0.0
    temporal_values = 0
    target_fraction = []
    previous_source = previous_removed = previous_mask = None
    for index, (source_frame, removed_frame, output_frame) in enumerate(
        zip(source, removed, output, strict=True)
    ):
        mask = cv2.imread(
            str(clip / "masks_final" / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE
        ) > 127
        background = ~mask
        difference = source_frame.astype(np.float32) - removed_frame.astype(np.float32)
        background_abs_sum += float(np.abs(difference[background]).sum())
        background_squared_sum += float(np.square(difference[background]).sum())
        background_values += int(background.sum()) * 3
        masked_abs_sum += float(np.abs(difference[mask]).sum())
        masked_values += int(mask.sum()) * 3
        render = cv2.imread(
            str(clip / "render_aligned" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED
        )
        target = render[..., 3] > 2
        target_fraction.append(float(target.mean()))
        safe = ~(mask | target)
        output_difference = removed_frame.astype(np.float32) - output_frame.astype(np.float32)
        output_background_abs_sum += float(np.abs(output_difference[safe]).sum())
        output_background_values += int(safe.sum()) * 3
        if previous_source is not None:
            common = background & ~previous_mask
            source_delta = source_frame.astype(np.float32) - previous_source.astype(np.float32)
            removed_delta = removed_frame.astype(np.float32) - previous_removed.astype(np.float32)
            temporal_abs_sum += float(np.abs(source_delta[common] - removed_delta[common]).sum())
            temporal_values += int(common.sum()) * 3
        previous_source, previous_removed, previous_mask = source_frame, removed_frame, mask
    mse = background_squared_sum / max(background_values, 1)
    method, removal_seconds = removal_runtime(clip)
    rendering_seconds = render_runtime(clip)
    clip_metadata = json.loads((clip / "clip.json").read_text())
    metrics: dict[str, float | int | str | bool] = {
        "dataset": clip.parent.name,
        "clip": clip.name,
        "episode": int(clip_metadata["episode"]),
        "task": str(clip_metadata["task"]),
        "published_task": str(clip_metadata["published_task"]),
        "frames": len(source),
        "fps": float(clip_metadata["fps"]),
        "frame_parity": True,
        "removal_method": method,
        "background_mae_removal": background_abs_sum / max(background_values, 1) / 255.0,
        "background_psnr_removal_db": 10.0 * math.log10(255.0**2 / max(mse, 1e-12)),
        "masked_region_change_mae": masked_abs_sum / max(masked_values, 1) / 255.0,
        "temporal_background_error": temporal_abs_sum / max(temporal_values, 1) / 255.0,
        "output_background_mae": output_background_abs_sum
        / max(output_background_values, 1)
        / 255.0,
        "target_alpha_fraction": float(np.mean(target_fraction)),
        "removal_seconds": removal_seconds,
        "render_seconds": rendering_seconds,
        "production_fps": len(source) / max(removal_seconds + rendering_seconds, 1e-12),
        "projection_calibrated": False,
    }
    metrics.update(kinematic_metrics(clip / "trajectory.npz"))
    return metrics


def main() -> None:
    rows = [evaluate_clip(clip) for clip in sorted(BENCHMARK.glob("*/*"))]
    numeric = [
        "background_mae_removal",
        "background_psnr_removal_db",
        "masked_region_change_mae",
        "temporal_background_error",
        "output_background_mae",
        "target_alpha_fraction",
        "removal_seconds",
        "render_seconds",
        "production_fps",
        "feasible_fraction",
        "position_error_p95_mm",
        "orientation_error_p95_rad",
    ]
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    aggregates = {
        dataset: {key: float(np.mean([float(row[key]) for row in values])) for key in numeric}
        for dataset, values in by_dataset.items()
    }
    aggregates["all"] = {
        key: float(np.mean([float(row[key]) for row in rows])) for key in numeric
    }
    payload = {
        "method": "OpenArm kinematic retargeting with camera-aware removal and deterministic render",
        "clips": rows,
        "aggregate": aggregates,
        "metric_scope": {
            "background": "pixels outside the audited source-robot mask",
            "temporal": "error between source and removal frame differences outside consecutive masks",
            "projection": "uncalibrated mask registration for this cross-dataset benchmark",
        },
    }
    (BENCHMARK / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (BENCHMARK / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
