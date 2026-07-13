from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..schema import Episode, SourceConfig


def load_agibot_h5(
    path: str | Path,
    config: SourceConfig,
    allow_uncalibrated: bool = False,
) -> Episode:
    if not config.calibrated and not allow_uncalibrated:
        raise ValueError("AgiBot base/flange calibration has not been validated")
    with h5py.File(path, "r") as data:
        raw_timestamp = np.asarray(data[config.fields.get("timestamp", "/timestamp")])
        # Subtract in integer nanoseconds before conversion. Casting epoch-scale nanoseconds
        # directly to float64 needlessly discards timing precision.
        timestamp = (raw_timestamp - raw_timestamp[0]).astype(np.float64) * 1e-9
        position = np.asarray(data[config.fields.get("position", "/state/end/position")])
        orientation = np.asarray(data[config.fields.get("orientation", "/state/end/orientation")])
        raw_gripper = np.asarray(data[config.fields.get("gripper", "/state/effector/position")])
    # AgiBot publishes left then right; canonical storage is right then left.
    raw_pose = np.concatenate([position, orientation], axis=-1)[:, [1, 0]]
    pose = np.stack(
        [
            config.pose_transform(side).apply(raw_pose[:, side_index])
            for side_index, side in enumerate(("right", "left"))
        ],
        axis=1,
    )
    gripper = raw_gripper[:, [1, 0]].astype(np.float64)
    if config.gripper_mode == "width_mm":
        low = np.nanpercentile(gripper, 1, axis=0)
        high = np.nanpercentile(gripper, 99, axis=0)
        # Published state is finger opening in millimetres: larger is more open.
        gripper = (high - gripper) / np.maximum(high - low, 1e-6)
    elif np.nanmax(gripper) > 1:
        gripper /= max(float(np.nanpercentile(gripper, 99)), 1e-6)
    episode = Episode(
        timestamp=timestamp,
        ee_pose=pose,
        gripper=np.clip(gripper, 0, 1),
        task=config.name,
        source_dataset=config.repo_id,
        source_episode=Path(path).parent.name,
        metadata={
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "active_sides": ["right", "left"],
        },
    )
    episode.validate()
    return episode


def load_agibot_lerobot_episode(
    path: str | Path | pa.Table,
    config: SourceConfig,
    episode_index: int,
    allow_uncalibrated: bool = False,
) -> Episode:
    """Load the provenance-labelled GT-111 AgiBot EE conversion."""
    if not config.calibrated and not allow_uncalibrated:
        raise ValueError("AgiBot base/flange calibration has not been validated")
    table = path if isinstance(path, pa.Table) else pq.read_table(path)
    mask = np.asarray(table["episode_index"]) == episode_index
    if not np.any(mask):
        raise KeyError(episode_index)
    timestamp = np.asarray(table["timestamp"])[mask].astype(np.float64)
    state = np.asarray(table[config.fields.get("pose", "observation.state")].to_pylist())[mask]
    pose = np.empty((len(timestamp), 2, 7), dtype=np.float64)
    gripper = np.empty((len(timestamp), 2), dtype=np.float64)
    # Source state is left xyz-wxyz-gripper, then right; canonical order is right, left.
    for source_index, target_index in ((0, 1), (1, 0)):
        offset = source_index * 8
        raw = np.concatenate(
            [
                state[:, offset : offset + 3],
                state[:, [offset + 4, offset + 5, offset + 6, offset + 3]],
            ],
            axis=1,
        )
        pose[:, target_index] = config.pose_transform(
            "left" if source_index == 0 else "right"
        ).apply(raw)
        gripper[:, target_index] = np.clip(state[:, offset + 7], 0, 1)
    result = Episode(
        timestamp=timestamp,
        ee_pose=pose,
        gripper=gripper,
        task="AgiBot World Alpha derived fallback",
        source_dataset=config.repo_id,
        source_episode=str(episode_index),
        metadata={
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "active_sides": ["right", "left"],
            "provenance_warning": "accessible third-party conversion, not gated original",
        },
    )
    result.validate()
    return result
