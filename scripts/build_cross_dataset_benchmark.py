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


SELECTIONS = (
    Selection(
        "agibot_world_alpha",
        649390,
        "water_pouring_demo_a",
        30,
        ROOT / "data/converted/agibot_openarm/episodes/episode_649390.npz",
        ROOT / "data/samples/agibot-world__AgiBotWorld-Alpha/sample/observations/410/649390",
    ),
    Selection(
        "agibot_world_alpha",
        649684,
        "water_pouring_demo_b",
        30,
        ROOT / "data/converted/agibot_openarm/episodes/episode_649684.npz",
        ROOT / "data/samples/agibot-world__AgiBotWorld-Alpha/sample/observations/410/649684",
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
    ),
    Selection(
        "molmoact2_tabletop",
        0,
        "close_box",
        20,
        ROOT / "data/converted/molmo_openarm/episodes/episode_000000.npz",
        ROOT / "data/samples/allenai__MolmoAct2-MolmoAct-Dataset-Tabletop",
        "observation.images.primary",
    ),
    Selection(
        "molmoact2_tabletop",
        460,
        "flip_mug_upright",
        20,
        ROOT / "data/converted/molmo_openarm/episodes/episode_000460.npz",
        ROOT / "data/samples/allenai__MolmoAct2-MolmoAct-Dataset-Tabletop",
        "observation.images.primary",
    ),
    Selection(
        "unifolm_wbt",
        0,
        "load_plates_demo_a",
        30,
        ROOT / "data/converted/unifolm_openarm/episodes/episode_000000.npz",
        ROOT / "data/samples/unitreerobotics__G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
        "observation.images.head_stereo_left",
    ),
    Selection(
        "unifolm_wbt",
        20,
        "load_plates_demo_b",
        30,
        ROOT / "data/converted/unifolm_openarm/episodes/episode_000020.npz",
        ROOT / "data/samples/unitreerobotics__G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
        "observation.images.head_stereo_left",
    ),
)


def _episode_rows(root: Path) -> dict[int, dict]:
    rows: list[dict] = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return {int(row["episode_index"]): row for row in rows}


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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    row_cache: dict[Path, dict[int, dict]] = {}
    manifest: list[dict] = []
    for selection in SELECTIONS:
        if not selection.converted.exists():
            raise FileNotFoundError(selection.converted)
        rows = None
        if selection.camera:
            if selection.source_root not in row_cache:
                row_cache[selection.source_root] = _episode_rows(selection.source_root)
            rows = row_cache[selection.source_root]
        with np.load(selection.converted, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        first, last = _motion_window(
            arrays["joint_position"], arrays["feasible"], 6 * selection.fps
        )
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
        sliced["timestamp"] = sliced["timestamp"] - sliced["timestamp"][0]
        sliced["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(destination / "trajectory.npz", **sliced)
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
