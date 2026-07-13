from __future__ import annotations

from dataclasses import replace

import numpy as np

from .constants import SIDES
from .gripper import preserve_pinch_center
from .poses import pose_to_matrix
from .schema import Episode, SourceConfig


def apply_registration(
    episode: Episode, source_config: SourceConfig, registration: dict
) -> Episode:
    """Apply a recorded registration to an already parsed source-frame episode."""
    config = replace(
        source_config,
        openarm_from_source_base=registration["openarm_from_source_base"],
        source_tool_from_openarm_tool=registration["source_tool_from_openarm_tool"],
        position_scale=float(registration.get("position_scale", source_config.position_scale)),
        calibrated=bool(registration.get("validated", False)),
    )
    active = tuple(episode.metadata.get("active_sides", SIDES))
    pose = episode.ee_pose.copy()
    for side in active:
        index = SIDES.index(side)
        pose[:, index] = config.pose_transform(side).apply(pose[:, index])
    diagnostics = {key: value.copy() for key, value in episode.diagnostics.items()}
    if config.preserve_pinch_center:
        offset = _source_pinch_offset_in_openarm_frame(config, episode.gripper)
        pose, diagnostics["pinch_center_target_m"] = preserve_pinch_center(
            pose, episode.gripper, offset
        )
    result = Episode(
        timestamp=episode.timestamp.copy(),
        ee_pose=pose,
        gripper=episode.gripper.copy(),
        task=episode.task,
        source_dataset=episode.source_dataset,
        source_episode=episode.source_episode,
        gripper_width_m=(
            episode.gripper_width_m.copy() if episode.gripper_width_m is not None else None
        ),
        diagnostics=diagnostics,
        metadata={
            **episode.metadata,
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "registration": registration,
            "pinch_center_compensated": config.preserve_pinch_center,
        },
    )
    result.validate()
    return result


def _source_pinch_offset_in_openarm_frame(
    config: SourceConfig, gripper: np.ndarray
) -> np.ndarray | None:
    if config.source_pinch_center_open_m is None or config.source_pinch_center_closed_m is None:
        return None

    def point(value: dict[str, list[float]] | list[float], side: str) -> np.ndarray:
        selected = value[side] if isinstance(value, dict) else value
        result = np.asarray(selected, dtype=np.float64)
        if result.shape != (3,):
            raise ValueError("source pinch centers must contain xyz triples")
        return result

    result = np.empty((*gripper.shape, 3), dtype=np.float64)
    for side_index, side in enumerate(SIDES):
        source_open = point(config.source_pinch_center_open_m, side)
        source_closed = point(config.source_pinch_center_closed_m, side)
        source_delta = (1 - gripper[:, side_index, None]) * (source_open - source_closed)
        source_from_openarm = pose_to_matrix(
            config.pose_transform(side).source_tool_from_openarm_tool
        )
        result[:, side_index] = source_delta @ source_from_openarm[:3, :3]
    return result
