#!/usr/bin/env python3
"""Export fast transparent OpenArm renders for every benchmark trajectory."""

from __future__ import annotations

import json
from pathlib import Path

from openarm_retarget.raytrace import export_blender_scene
from openarm_retarget.schema import Episode


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "outputs/cross_dataset_openarm_benchmark"


def main() -> None:
    for clip in sorted(BENCHMARK.glob("*/*")):
        trajectory = clip / "trajectory.npz"
        if not trajectory.exists():
            continue
        episode = Episode.load(trajectory)
        scene_path = export_blender_scene(
            episode,
            clip / "render_scene",
            width=640,
            height=480,
            samples=0,
            eevee_samples=16,
            png_compression=15,
        )
        scene = json.loads(scene_path.read_text())
        active = set(episode.metadata.get("active_sides", ["right", "left"]))
        scene["objects"] = [
            item
            for item in scene["objects"]
            if not any(f"_{side}_" in item["name"] for side in {"right", "left"} - active)
        ]
        scene["benchmark_projection"] = {
            "mode": "uncalibrated preview followed by source-mask affine registration",
            "active_sides": sorted(active),
            "warning": "Image projection is not a measured source-camera calibration.",
        }
        scene_path.write_text(json.dumps(scene, separators=(",", ":")) + "\n")
        if active == {"right", "left"}:
            for side in ("left", "right"):
                side_scene = dict(scene)
                side_scene["objects"] = [
                    item for item in scene["objects"] if f"_{side}_" in item["name"]
                ]
                side_scene["benchmark_projection"] = {
                    **scene["benchmark_projection"],
                    "rendered_side": side,
                }
                (scene_path.parent / f"scene_{side}.json").write_text(
                    json.dumps(side_scene, separators=(",", ":")) + "\n"
                )
        print(scene_path)


if __name__ == "__main__":
    main()
