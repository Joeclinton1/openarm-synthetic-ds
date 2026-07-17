from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .constants import SIDES
from .gripper import preserve_pinch_center
from .ik import OpenArmIK
from .schema import Episode, SourceConfig


def integrate_planar_base_velocity(timestamp: np.ndarray, velocity_xy: np.ndarray) -> np.ndarray:
    """Integrate source-frame floor velocity while fixing base height and orientation."""
    timestamp = np.asarray(timestamp, dtype=np.float64)
    velocity_xy = np.asarray(velocity_xy, dtype=np.float64)
    if velocity_xy.shape != (len(timestamp), 2):
        raise ValueError("velocity_xy must have shape [T, 2]")
    translation = np.zeros((len(timestamp), 3), dtype=np.float64)
    if len(timestamp) > 1:
        dt = np.diff(timestamp)
        if np.any(dt <= 0):
            raise ValueError("timestamps must be strictly increasing")
        increments = 0.5 * (velocity_xy[:-1] + velocity_xy[1:]) * dt[:, None]
        translation[1:, :2] = np.cumsum(increments, axis=0)
    return translation


def load_hiw_episode(
    parquet_path: str | Path | pa.Table,
    config: SourceConfig,
    episode_index: int,
    allow_uncalibrated: bool = False,
    model_path: str | Path | None = None,
) -> Episode:
    if not config.calibrated and not allow_uncalibrated:
        raise ValueError("HIW-500 G1-root to OpenArm-root calibration is required")
    table = parquet_path if isinstance(parquet_path, pa.Table) else pq.read_table(parquet_path)
    mask = np.asarray(table["episode_index"]) == episode_index
    if not np.any(mask):
        raise KeyError(episode_index)
    timestamp = np.asarray(table["timestamp"])[mask].astype(np.float64)
    wbc = np.asarray(table["observation.state.wbc"].to_pylist(), dtype=np.float64)[mask]
    poses = np.zeros((len(timestamp), 2, 7), dtype=np.float64)
    gripper = np.zeros((len(timestamp), 2), dtype=np.float64)
    for source_index, side in enumerate(("left", "right")):
        target = SIDES.index(side)
        offset = 7 + 6 * source_index
        raw = np.concatenate(
            [
                wbc[:, offset : offset + 3],
                Rotation.from_euler("xyz", wbc[:, offset + 3 : offset + 6]).as_quat(),
            ],
            axis=1,
        )
        poses[:, target] = config.pose_transform(side).apply(raw)
        # Published squeeze is at indices 20 and 22, already 0=open, 1=closed.
        gripper[:, target] = np.clip(wbc[:, 20 + 2 * source_index], 0, 1)
    diagnostics = {
        "mobile_base_translation_m": integrate_planar_base_velocity(timestamp, wbc[:, :2])
    }
    if config.preserve_pinch_center:
        poses, diagnostics["pinch_center_target_m"] = preserve_pinch_center(poses, gripper)
    result = Episode(
        timestamp=timestamp,
        ee_pose=poses,
        gripper=gripper,
        task="HIW-500",
        source_dataset=config.repo_id,
        source_episode=str(episode_index),
        diagnostics=diagnostics,
        metadata={
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "active_sides": ["right", "left"],
            "pinch_center_compensated": config.preserve_pinch_center,
            "gripper_calibration": "normalized_only",
            "mobile_base_model": {
                "translation_velocity_fields": ["pivot_vx", "pivot_vy"],
                "floor_plane_axes": ["source_x", "source_y"],
                "fixed_vertical_axis": "source_z",
                "fixed_height": True,
                "fixed_orientation": True,
                "ignored_fields": [
                    "pivot_vyaw",
                    "pivot_roll",
                    "pivot_pitch",
                    "pivot_yaw",
                    "pivot_height",
                ],
            },
        },
    )
    result.validate()
    return result


def fit_workspace_translation(
    episode: Episode,
    model_path: str | Path | None = None,
    sides: tuple[str, ...] = SIDES,
) -> dict[str, list[float] | str]:
    """Fit translation only, preserving documented source axes and metric scale.

    This is an inspection bootstrap, not a metrological calibration. A real conversion must
    validate it using corresponding physical points or robot CAD transforms.
    """
    ik = OpenArmIK(model_path)
    indices = [SIDES.index(side) for side in sides]
    source_centres = np.nanmedian(episode.ee_pose[:, indices, :3], axis=0)
    target_centres = np.stack([ik.forward_pose(side, ik.neutral(side))[:3] for side in sides])
    translation = np.mean(target_centres - source_centres, axis=0)
    return {
        "method": "median_workspace_translation_unvalidated",
        "openarm_from_source_base": [*translation.tolist(), 0.0, 0.0, 0.0, 1.0],
        "source_tool_from_openarm_tool": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
