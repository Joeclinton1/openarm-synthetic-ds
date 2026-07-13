from __future__ import annotations

from dataclasses import replace

from .constants import SIDES
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
    result = Episode(
        timestamp=episode.timestamp.copy(),
        ee_pose=pose,
        gripper=episode.gripper.copy(),
        task=episode.task,
        source_dataset=episode.source_dataset,
        source_episode=episode.source_episode,
        metadata={
            **episode.metadata,
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "registration": registration,
        },
    )
    result.validate()
    return result
