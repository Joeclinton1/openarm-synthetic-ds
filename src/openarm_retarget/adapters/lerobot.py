from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from ..constants import SIDES
from ..gripper import preserve_pinch_center
from ..ik import OpenArmIK
from ..poses import convert_quaternion_order
from ..schema import Episode, SourceConfig


def _rotation(values: np.ndarray, config: SourceConfig) -> np.ndarray:
    representation = config.rotation_representation
    if representation == "quaternion":
        return convert_quaternion_order(values, config.quaternion_order)
    if representation == "euler":
        return Rotation.from_euler(config.rotation_euler_order, values).as_quat()
    if representation == "rotvec":
        return Rotation.from_rotvec(values).as_quat()
    raise ValueError(f"Unsupported rotation representation: {representation}")


def _pose_from_six(values: np.ndarray, config: SourceConfig) -> np.ndarray:
    return np.concatenate([values[:, :3], _rotation(values[:, 3:], config)], axis=1)


def _gripper(values: np.ndarray, config: SourceConfig) -> np.ndarray:
    if config.gripper_mode == "normalized":
        return np.clip(values, 0, 1)
    if config.gripper_mode == "hiw_trigger_squeeze":
        # Source trigger is 0..10 and squeeze is 0..1; squeeze has the usable close signal.
        return np.clip(values, 0, 1)
    if config.gripper_mode == "signed":
        low, high = np.nanpercentile(values, [1, 99])
        return np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    if config.gripper_mode == "mean_fingers":
        return np.clip(np.mean(values, axis=-1), 0, 1)
    if config.gripper_mode == "brainco_fingers":
        if values.shape[-1] != 6:
            raise ValueError("BrainCo hand must contain six values")
        return np.clip(np.mean(values[..., [0, 2, 3, 4, 5]], axis=-1), 0, 1)
    raise ValueError(f"Unsupported gripper mode: {config.gripper_mode}")


def load_lerobot_episode(
    parquet_path: str | Path | pa.Table,
    config: SourceConfig,
    episode_index: int,
    allow_uncalibrated: bool = False,
    model_path: str | Path | None = None,
) -> Episode:
    if not config.calibrated and not allow_uncalibrated:
        raise ValueError(
            f"{config.name} has no validated source-base/tool calibration; "
            "supply one in its config or explicitly allow uncalibrated inspection"
        )
    table = parquet_path if isinstance(parquet_path, pa.Table) else pq.read_table(parquet_path)
    table = table.filter(np.asarray(table["episode_index"]) == episode_index)
    if table.num_rows == 0:
        raise KeyError(f"Episode {episode_index} not present in source table")
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)
    pose_values = np.asarray(table[config.fields["pose"]].to_pylist(), dtype=np.float64)
    poses = np.empty((len(timestamp), 2, 7), dtype=np.float64)
    grippers = np.zeros((len(timestamp), 2), dtype=np.float64)
    ik = OpenArmIK(model_path)
    for side_index, side in enumerate(SIDES):
        poses[:, side_index] = ik.forward_pose(side, ik.neutral(side))

    if config.single_arm_side:
        target = SIDES.index(config.single_arm_side)
        poses[:, target] = config.pose_transform(config.single_arm_side).apply(
            _pose_from_six(pose_values[:, :6], config)
        )
        if "gripper" in config.fields:
            values = np.asarray(table[config.fields["gripper"]].to_pylist(), dtype=np.float64)
            if values.ndim > 1:
                values = values[:, int(config.fields.get("gripper_index", 0))]
            grippers[:, target] = _gripper(values, config)
    else:
        if pose_values.shape[1] != 12:
            raise ValueError("Bimanual Cartesian field must contain two 6D poses")
        for source_index, side in enumerate(config.arm_order):
            target_index = SIDES.index(side)
            poses[:, target_index] = config.pose_transform(side).apply(
                _pose_from_six(pose_values[:, source_index * 6 : (source_index + 1) * 6], config)
            )
        if "gripper" in config.fields:
            values = np.asarray(table[config.fields["gripper"]].to_pylist(), dtype=np.float64)
            if values.shape[1] == 2:
                for source_index, side in enumerate(config.arm_order):
                    grippers[:, SIDES.index(side)] = _gripper(values[:, source_index], config)
            elif values.shape[1] % 2 == 0:
                width = values.shape[1] // 2
                for source_index, side in enumerate(config.arm_order):
                    grippers[:, SIDES.index(side)] = _gripper(
                        values[:, source_index * width : (source_index + 1) * width], config
                    )
    diagnostics = {}
    if config.preserve_pinch_center:
        poses, diagnostics["pinch_center_target_m"] = preserve_pinch_center(poses, grippers)
    episode = Episode(
        timestamp=timestamp,
        ee_pose=poses,
        gripper=grippers,
        task=config.name,
        source_dataset=config.repo_id,
        source_episode=str(episode_index),
        diagnostics=diagnostics,
        metadata={
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "active_sides": [config.single_arm_side] if config.single_arm_side else list(SIDES),
            "pinch_center_compensated": config.preserve_pinch_center,
            "gripper_calibration": "normalized_only",
        },
    )
    episode.validate()
    return episode
