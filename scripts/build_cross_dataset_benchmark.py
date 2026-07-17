#!/usr/bin/env python3
"""Prepare action-centred source/trajectory pairs for the visual benchmark."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openarm_retarget.camera import write_agibot_openarm_camera, write_static_openarm_camera
from openarm_retarget.presets import integrate_planar_base_velocity


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "cross_dataset_openarm_benchmark"


@dataclass(frozen=True)
class Selection:
    dataset: str
    episode: int
    task: str
    fps: int
    converted: Path
    source_root: Path
    camera: str | None = None
    crop: str | None = None
    fixed_start_frame: int | None = None


SELECTIONS = (
    Selection(
        "agibot_world_alpha",
        649390,
        "water_pouring_demo_a",
        30,
        ROOT / "data/converted/agibot_openarm/episodes/episode_649390.npz",
        ROOT / "data/samples/agibot-world__AgiBotWorld-Alpha/sample/observations/410/649390",
        fixed_start_frame=666,
    ),
    Selection(
        "agibot_world_alpha",
        649684,
        "water_pouring_demo_b",
        30,
        ROOT / "data/converted/agibot_openarm/episodes/episode_649684.npz",
        ROOT / "data/samples/agibot-world__AgiBotWorld-Alpha/sample/observations/410/649684",
        fixed_start_frame=805,
    ),
    Selection(
        "hiw_500",
        7,
        "hang_hanger",
        30,
        ROOT / "data/converted/hiw_openarm/episodes/episode_000007.npz",
        ROOT / "data/samples/BitRobot__HIW-500-LeRobot",
        "observation.images.head",
        "crop=640:480:0:0",
        fixed_start_frame=374,
    ),
    Selection(
        "hiw_500",
        10,
        "hang_keys_on_hook",
        30,
        ROOT / "data/converted/hiw_openarm/episodes/episode_000010.npz",
        ROOT / "data/samples/BitRobot__HIW-500-LeRobot",
        "observation.images.head",
        "crop=640:480:0:0",
        fixed_start_frame=474,
    ),
    Selection(
        "molmoact2_tabletop",
        0,
        "close_box",
        20,
        ROOT / "data/converted/molmo_openarm/episodes/episode_000000.npz",
        ROOT / "data/samples/allenai__MolmoAct2-MolmoAct-Dataset-Tabletop",
        "observation.images.primary",
        fixed_start_frame=36,
    ),
    Selection(
        "molmoact2_tabletop",
        460,
        "flip_mug_upright",
        20,
        ROOT / "data/converted/molmo_openarm/episodes/episode_000460.npz",
        ROOT / "data/samples/allenai__MolmoAct2-MolmoAct-Dataset-Tabletop",
        "observation.images.primary",
        fixed_start_frame=415,
    ),
    Selection(
        "droid",
        13,
        "wipe_table_with_cloth",
        15,
        ROOT / "data/converted/droid_openarm/episodes/episode_000013.npz",
        ROOT / "data/samples/lerobot__droid_1.0.1",
        "observation.images.exterior_1_left",
        fixed_start_frame=252,
    ),
    Selection(
        "droid",
        55,
        "pour_into_two_bowls",
        15,
        ROOT / "data/converted/droid_openarm/episodes/episode_000055.npz",
        ROOT / "data/samples/lerobot__droid_1.0.1",
        "observation.images.exterior_1_left",
        fixed_start_frame=180,
    ),
    Selection(
        "rh20t_franka",
        35,
        "unscrew_jar_lid",
        10,
        ROOT / "data/converted/rh20t_franka_openarm/episodes/episode_000035.npz",
        ROOT / "data/samples/robot-lev__rh20t_cfg5",
        "observation.images.cam_037522062165",
        fixed_start_frame=392,
    ),
    Selection(
        "rh20t_franka",
        73,
        "put_knife_on_rack",
        10,
        ROOT / "data/converted/rh20t_franka_openarm/episodes/episode_000073.npz",
        ROOT / "data/samples/robot-lev__rh20t_cfg5",
        "observation.images.cam_037522062165",
        fixed_start_frame=265,
    ),
    Selection(
        "robomind_agilex_3rgb",
        16,
        "load_plate_rack",
        30,
        ROOT / "data/converted/robomind_agilex_openarm/episodes/episode_000016.npz",
        ROOT / "data/samples/Traly__RoboMIND-lerobot/agilex_3rgb",
        "observation.images.camera_front",
        fixed_start_frame=337,
    ),
    Selection(
        "robomind_agilex_3rgb",
        215,
        "clean_cup_with_brush",
        30,
        ROOT
        / "data/converted/robomind_agilex_benchmark_supplement/episodes/episode_000215.npz",
        ROOT / "data/samples/Traly__RoboMIND-lerobot/agilex_3rgb",
        "observation.images.camera_front",
        fixed_start_frame=177,
    ),
)


def _episode_rows(root: Path, camera: str) -> dict[int, dict]:
    columns = [
        "episode_index",
        "tasks",
        f"videos/{camera}/chunk_index",
        f"videos/{camera}/file_index",
        f"videos/{camera}/from_timestamp",
    ]
    rows: list[dict] = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return {int(row["episode_index"]): row for row in rows}


def _hiw_mobile_base_translations(root: Path, episodes: set[int]) -> dict[int, np.ndarray]:
    table = pq.read_table(
        root / "sample/data.parquet",
        columns=["episode_index", "timestamp", "observation.state.wbc"],
    )
    episode_column = np.asarray(table["episode_index"])
    timestamp_column = np.asarray(table["timestamp"], dtype=np.float64)
    wbc_column = np.asarray(table["observation.state.wbc"].to_pylist(), dtype=np.float64)
    result = {}
    for episode in episodes:
        selected = episode_column == episode
        if not np.any(selected):
            raise KeyError(f"HIW sample does not contain episode {episode}")
        result[episode] = integrate_planar_base_velocity(
            timestamp_column[selected], wbc_column[selected, :2]
        )
    return result


def _motion_window(joints: np.ndarray, feasible: np.ndarray, frames: int) -> tuple[int, int]:
    count = len(joints)
    frames = min(frames, count)
    if frames == count:
        return 0, count
    velocity = np.linalg.norm(np.diff(joints, axis=0), axis=(1, 2))
    score = np.convolve(velocity, np.ones(frames - 1), mode="valid")
    # Strongly discourage windows dominated by invalid IK frames.
    validity = np.convolve(feasible.astype(float), np.ones(frames), mode="valid") / frames
    score = score * np.square(validity[: len(score)])
    start = int(np.argmax(score))
    return start, start + frames


def _moving_joint_sides(joints: np.ndarray, declared: list[str]) -> list[str]:
    sides = ("right", "left")
    motion = np.linalg.norm(np.diff(joints, axis=0), axis=2).sum(axis=0)
    maximum = float(np.max(motion))
    if maximum < 1e-6:
        return declared
    moving = [
        side
        for index, side in enumerate(sides)
        if side in declared and float(motion[index]) >= max(1e-3, 0.05 * maximum)
    ]
    return moving or declared


def _slug(selection: Selection) -> str:
    task = re.sub(r"[^a-z0-9]+", "_", selection.task.lower()).strip("_")
    return f"{task}__episode_{selection.episode:06d}"


def _source_location(selection: Selection, rows: dict[int, dict] | None) -> tuple[Path, int]:
    if selection.camera is None:
        return selection.source_root / "videos/head_color.mp4", 0
    assert rows is not None
    row = rows[selection.episode]
    chunk = int(row[f"videos/{selection.camera}/chunk_index"])
    file_index = int(row[f"videos/{selection.camera}/file_index"])
    timestamp = float(row[f"videos/{selection.camera}/from_timestamp"])
    source = (
        selection.source_root
        / "videos"
        / selection.camera
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    return source, round(timestamp * selection.fps)


def _write_agibot_camera(
    selection: Selection,
    destination: Path,
    first: int,
    last: int,
    output_video: Path,
) -> None:
    """Slice source camera poses to the exact benchmark window and map them to OpenArm."""
    sample_root = selection.source_root.parents[2]
    camera_root = (
        sample_root / "parameters" / "410" / str(selection.episode) / "camera"
    )
    extrinsics = json.loads(
        (camera_root / "head_extrinsic_params_aligned.json").read_text()
    )
    sliced_extrinsics = destination / "source_camera_extrinsics.json"
    sliced_extrinsics.write_text(json.dumps(extrinsics[first:last], separators=(",", ":")) + "\n")
    write_agibot_openarm_camera(
        destination / "trajectory.npz",
        camera_root / "head_intrinsic_params.json",
        sliced_extrinsics,
        destination / "camera.json",
        video_path=output_video,
        calibration_width=1280,
        calibration_height=720,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    row_cache: dict[Path, dict[int, dict]] = {}
    hiw_mobile_cache: dict[int, np.ndarray] | None = None
    manifest: list[dict] = []
    for selection in SELECTIONS:
        if not selection.converted.exists():
            raise FileNotFoundError(selection.converted)
        rows = None
        if selection.camera:
            if selection.source_root not in row_cache:
                row_cache[selection.source_root] = _episode_rows(
                    selection.source_root, selection.camera
                )
            rows = row_cache[selection.source_root]
        with np.load(selection.converted, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if (
            selection.dataset == "hiw_500"
            and "diagnostic_mobile_base_translation_m" not in arrays
        ):
            if hiw_mobile_cache is None:
                hiw_mobile_cache = _hiw_mobile_base_translations(
                    selection.source_root,
                    {item.episode for item in SELECTIONS if item.dataset == "hiw_500"},
                )
            arrays["diagnostic_mobile_base_translation_m"] = hiw_mobile_cache[
                selection.episode
            ]
        first, last = _motion_window(
            arrays["joint_position"], arrays["feasible"], 6 * selection.fps
        )
        if selection.fixed_start_frame is not None:
            first = selection.fixed_start_frame
            last = min(first + 6 * selection.fps, len(arrays["timestamp"]))
        destination = OUTPUT / selection.dataset / _slug(selection)
        destination.mkdir(parents=True, exist_ok=True)
        source, episode_offset = _source_location(selection, rows)
        output_video = destination / "01_source.mp4"
        filters = [
            f"trim=start_frame={episode_offset + first}:end_frame={episode_offset + last}",
            "setpts=PTS-STARTPTS",
        ]
        if selection.crop:
            filters.append(selection.crop)
        filters.append("scale=640:480:flags=lanczos")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-an",
                "-vf",
                ",".join(filters),
                "-r",
                str(selection.fps),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                str(output_video),
            ],
            check=True,
        )
        sliced = {}
        for key, value in arrays.items():
            sliced[key] = value[first:last] if value.ndim and len(value) == len(arrays["timestamp"]) else value
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata.update(
            {
                "benchmark_task": selection.task,
                "benchmark_first_frame": first,
                "benchmark_last_frame_exclusive": last,
                "benchmark_source_video": str(source.relative_to(ROOT)),
            }
        )
        if selection.dataset == "hiw_500":
            metadata["mobile_base_model"] = {
                "translation_velocity_fields": ["pivot_vx", "pivot_vy"],
                "floor_plane_axes": ["source_x", "source_y"],
                "fixed_vertical_axis": "source_z",
                "fixed_height": True,
                "fixed_orientation": True,
                "shared_translation_for_both_shoulders": True,
                "ignored_fields": [
                    "pivot_vyaw",
                    "pivot_roll",
                    "pivot_pitch",
                    "pivot_yaw",
                    "pivot_height",
                ],
            }
        metadata["active_sides"] = _moving_joint_sides(
            sliced["joint_position"], metadata.get("active_sides", ["right", "left"])
        )
        sliced["timestamp"] = sliced["timestamp"] - sliced["timestamp"][0]
        sliced["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(destination / "trajectory.npz", **sliced)
        if selection.dataset == "agibot_world_alpha":
            _write_agibot_camera(selection, destination, first, last, output_video)
        elif selection.dataset == "molmoact2_tabletop":
            write_static_openarm_camera(
                destination / "trajectory.npz",
                ROOT / "configs/cameras/molmoact2_tabletop_fitted.json",
                destination / "camera.json",
            )
        record = {
            "dataset": selection.dataset,
            "episode": selection.episode,
            "task": selection.task,
            "published_task": (rows[selection.episode]["tasks"][0] if rows else "Water Pouring in Restaurant"),
            "fps": selection.fps,
            "frames": last - first,
            "episode_first_frame": first,
            "episode_last_frame_exclusive": last,
            "source": str(output_video.relative_to(OUTPUT)),
            "trajectory": str((destination / "trajectory.npz").relative_to(OUTPUT)),
            "calibration_validated": False,
        }
        (destination / "clip.json").write_text(json.dumps(record, indent=2) + "\n")
        manifest.append(record)
        print(f"prepared {destination.relative_to(OUTPUT)}: frames {first}:{last}")
    (OUTPUT / "manifest.json").write_text(json.dumps({"clips": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
