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
            camera_json=(clip / "camera.json") if (clip / "camera.json").exists() else None,
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
        camera_registered = (
            clip.parent.name == "agibot_world_alpha" and (clip / "camera.json").exists()
        )
        camera_fitted = (
            clip.parent.name == "molmoact2_tabletop" and (clip / "camera.json").exists()
        )
        scene["benchmark_projection"] = {
            "mode": (
                "accepted AgiBot fixture camera registration"
                if camera_registered
                else "audited Molmo fixed-camera fit"
                if camera_fitted
                else "uncalibrated preview followed by source-mask affine registration"
            ),
            "active_sides": sorted(active),
            "warning": (
                "Source camera mapping reproduced on the complete AgiBot episode 649684 fixture."
                if camera_registered
                else "Fixed camera estimated from 240 audited source end-effector correspondences."
                if camera_fitted
                else "Image projection is not a measured source-camera calibration."
            ),
        }
        scene_path.write_text(json.dumps(scene, separators=(",", ":")) + "\n")
        for side in ("left", "right"):
            (scene_path.parent / f"scene_{side}.json").unlink(missing_ok=True)
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
