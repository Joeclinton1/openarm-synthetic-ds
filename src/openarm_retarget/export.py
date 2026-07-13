from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .constants import FPS, JOINT_NAMES, POSE_NAMES, SIDES
from .gripper import closure_to_finger_qpos
from .schema import Episode


def _fixed(values: np.ndarray) -> pa.Array:
    values = np.asarray(values)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def _joint_vector(episode: Episode) -> np.ndarray:
    if episode.joint_position is None:
        raise ValueError("Episode has no OpenArm joint solution")
    n = len(episode.timestamp)
    result = np.empty((n, 16), dtype=np.float32)
    fingers = closure_to_finger_qpos(episode.gripper)
    for side_index, side in enumerate(SIDES):
        output = side_index * 8
        result[:, output : output + 7] = episode.joint_position[:, side_index]
        result[:, output + 7] = fingers[:, side_index * 2]
    return result


def _statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).reshape(-1).tolist(),
        "max": np.max(values, axis=0).reshape(-1).tolist(),
        "mean": np.mean(values, axis=0).reshape(-1).tolist(),
        "std": np.std(values, axis=0).reshape(-1).tolist(),
        "count": [len(values)],
        **{
            f"q{int(quantile * 100):02d}": np.quantile(values, quantile, axis=0)
            .reshape(-1)
            .tolist()
            for quantile in (0.01, 0.10, 0.50, 0.90, 0.99)
        },
    }


def export_lerobot_v3(
    episodes: list[Episode],
    destination: str | Path,
    fps: int = FPS,
    feasible_only: bool = True,
    allow_uncalibrated: bool = False,
) -> Path:
    if not allow_uncalibrated:
        uncalibrated = [
            episode.source_episode for episode in episodes if not episode.metadata.get("calibrated")
        ]
        if uncalibrated:
            raise ValueError(
                "Refusing to export uncalibrated episodes: " + ", ".join(uncalibrated[:5])
            )
    destination = Path(destination)
    data_dir = destination / "data/chunk-000"
    meta_episode_dir = destination / "meta/episodes/chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_episode_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pa.Table] = []
    episode_meta: list[dict] = []
    tasks: list[str] = []
    global_index = 0
    output_episode_index = 0
    for episode in episodes:
        episode.validate()
        joint = _joint_vector(episode)
        keep = (
            episode.feasible
            if feasible_only and episode.feasible is not None
            else np.ones(len(joint), bool)
        )
        # Keep complete feasible runs as separate episodes so timestamps remain contiguous.
        boundaries = np.flatnonzero(np.diff(np.r_[False, keep, False]))
        for start, end in boundaries.reshape(-1, 2):
            start, end = int(start), int(end)
            if end <= start:
                continue
            task_index = tasks.index(episode.task) if episode.task in tasks else len(tasks)
            if task_index == len(tasks):
                tasks.append(episode.task)
            count = int(end - start)
            timestamps = np.arange(count, dtype=np.float64) / fps
            right_pose = episode.ee_pose[start:end, 0].astype(np.float32)
            left_pose = episode.ee_pose[start:end, 1].astype(np.float32)
            table = pa.table(
                {
                    "action": _fixed(joint[start:end]),
                    "observation.state": _fixed(joint[start:end]),
                    "observation.ee_pose.right": _fixed(right_pose),
                    "observation.ee_pose.left": _fixed(left_pose),
                    "timestamp": timestamps,
                    "frame_index": np.arange(count, dtype=np.int64),
                    "episode_index": np.full(count, output_episode_index, dtype=np.int64),
                    "index": np.arange(global_index, global_index + count, dtype=np.int64),
                    "task_index": np.full(count, task_index, dtype=np.int64),
                    "retarget.position_error_m": _fixed(
                        episode.diagnostics.get("position_error_m", np.zeros((len(joint), 2)))[
                            start:end
                        ].astype(np.float32)
                    ),
                    "retarget.orientation_error_rad": _fixed(
                        episode.diagnostics.get("orientation_error_rad", np.zeros((len(joint), 2)))[
                            start:end
                        ].astype(np.float32)
                    ),
                }
            )
            rows.append(table)
            episode_meta.append(
                {
                    "episode_index": output_episode_index,
                    "tasks": [episode.task],
                    "length": count,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_index,
                    "dataset_to_index": global_index + count,
                    "source_dataset": episode.source_dataset,
                    "source_episode": episode.source_episode,
                }
            )
            global_index += count
            output_episode_index += 1
    if not rows:
        raise ValueError("No feasible frames to export")
    data = pa.concat_tables(rows)
    pq.write_table(data, data_dir / "file-000.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(episode_meta), meta_episode_dir / "file-000.parquet")
    pq.write_table(
        pa.table({"task_index": np.arange(len(tasks), dtype=np.int64), "task": tasks}),
        destination / "meta/tasks.parquet",
    )
    features = {
        "action": {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        **{
            f"observation.ee_pose.{side}": {"dtype": "float32", "shape": [7], "names": POSE_NAMES}
            for side in SIDES
        },
        "retarget.position_error_m": {"dtype": "float32", "shape": [2], "names": list(SIDES)},
        "retarget.orientation_error_rad": {"dtype": "float32", "shape": [2], "names": list(SIDES)},
        "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "openarm_bimanual_v2.0",
        "total_episodes": output_episode_index,
        "total_frames": global_index,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{output_episode_index}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "retargeting": {
            "pose_frame": "official OpenArm v2 MuJoCo root -> openarm_{side}_ee_base_link",
            "position_unit": "metre",
            "quaternion_order": "xyzw",
            "calibration_validated": not allow_uncalibrated,
        },
    }
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    statistics = {}
    for name in features:
        column = data[name]
        if pa.types.is_fixed_size_list(column.type):
            values = np.asarray(column.to_pylist())
        else:
            values = np.asarray(column)
        statistics[name] = _statistics(values)
    (destination / "meta/stats.json").write_text(json.dumps(statistics, indent=2) + "\n")
    return destination


def validate_lerobot_v3(destination: str | Path) -> dict:
    destination = Path(destination)
    info = json.loads((destination / "meta/info.json").read_text())
    data_files = sorted(destination.glob("data/chunk-*/*.parquet"))
    episode_files = sorted(destination.glob("meta/episodes/chunk-*/*.parquet"))
    errors: list[str] = []
    if not data_files:
        errors.append("no data parquet files")
        return {"ok": False, "errors": errors}
    data = pa.concat_tables([pq.read_table(path) for path in data_files])
    episodes = pa.concat_tables([pq.read_table(path) for path in episode_files])
    if data.num_rows != int(info["total_frames"]):
        errors.append("total_frames does not match parquet rows")
    if episodes.num_rows != int(info["total_episodes"]):
        errors.append("total_episodes does not match episode metadata")
    if set(data.column_names) != set(info["features"]):
        errors.append("info features do not exactly match parquet columns")
    index = np.asarray(data["index"])
    if not np.array_equal(index, np.arange(len(index))):
        errors.append("global index is not contiguous")
    for row in episodes.to_pylist():
        mask = np.asarray(data["episode_index"]) == int(row["episode_index"])
        frames = np.asarray(data["frame_index"])[mask]
        timestamps = np.asarray(data["timestamp"])[mask]
        if len(frames) != int(row["length"]):
            errors.append(f"episode {row['episode_index']} length mismatch")
        if not np.array_equal(frames, np.arange(len(frames))):
            errors.append(f"episode {row['episode_index']} frame index is not contiguous")
        if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 1 / info["fps"]):
            errors.append(f"episode {row['episode_index']} timestamps do not match fps")
    if not (destination / "meta/stats.json").is_file():
        errors.append("meta/stats.json is missing")
    return {
        "ok": not errors,
        "destination": str(destination),
        "total_frames": data.num_rows,
        "total_episodes": episodes.num_rows,
        "errors": errors,
    }
