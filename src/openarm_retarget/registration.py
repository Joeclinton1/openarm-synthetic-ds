from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import SIDES
from .ik import OpenArmIK
from .poses import matrix_to_pose, pose_to_matrix
from .schema import Episode, PoseConfig


def _mean_pose(poses: np.ndarray) -> np.ndarray:
    position = np.nanmedian(poses[:, :3], axis=0)
    rotation = Rotation.from_quat(poses[:, 3:]).mean().as_quat()
    return np.concatenate([position, rotation])


def _align_vector_with_up(source: np.ndarray, target: np.ndarray) -> Rotation:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    # The public sources and OpenArm model are z-up. A low-weight second vector fixes the
    # otherwise unconstrained roll around the bimanual right-to-left axis.
    source_vectors = np.stack([source, [0.0, 0.0, 1.0]])
    target_vectors = np.stack([target, [0.0, 0.0, 1.0]])
    rotation, _ = Rotation.align_vectors(target_vectors, source_vectors, weights=[1.0, 0.25])
    return rotation


def auto_register_episode(
    episode: Episode,
    model_path: str | Path | None = None,
    minimum_scale: float = 0.4,
    maximum_scale: float = 1.0,
    source_tool_from_openarm_tool: PoseConfig = None,
) -> dict:
    """Estimate an inspection-grade source-to-OpenArm workspace registration.

    The mapping aligns the median bimanual work pose to a reachable OpenArm posture, preserves
    relative SE(3) motion, and derives separate left/right tool-axis offsets. It is useful for
    automatic retargeting but is not a metrological camera/scene calibration.
    """
    episode.validate()
    solver = OpenArmIK(model_path)
    active = tuple(episode.metadata.get("active_sides", SIDES))

    def tool_prior(side: str) -> np.ndarray | None:
        if source_tool_from_openarm_tool is None:
            return None
        value = source_tool_from_openarm_tool
        if isinstance(value, dict):
            value = value[side]
        return np.asarray(value, dtype=np.float64)

    source_poses = {side: episode.ee_pose[:, SIDES.index(side)] for side in active}
    workspace_poses = {}
    for side in active:
        poses = source_poses[side]
        prior = tool_prior(side)
        if prior is not None:
            prior_matrix = pose_to_matrix(prior)
            poses = np.stack(
                [matrix_to_pose(pose_to_matrix(pose) @ prior_matrix) for pose in poses]
            )
        workspace_poses[side] = poses
    workspace_mean = {side: _mean_pose(workspace_poses[side]) for side in active}
    target_mean = {side: solver.forward_pose(side, solver.neutral(side)) for side in active}

    motion_radius = max(
        float(
            np.nanpercentile(
                np.linalg.norm(workspace_poses[side][:, :3] - workspace_mean[side][:3], axis=1), 95
            )
        )
        for side in active
    )
    scale = float(np.clip(0.20 / max(motion_radius, 1e-6), minimum_scale, maximum_scale))
    source_mean = {}
    for side in active:
        poses = source_poses[side].copy()
        poses[:, :3] *= scale
        prior = tool_prior(side)
        if prior is not None:
            prior_matrix = pose_to_matrix(prior)
            poses = np.stack(
                [matrix_to_pose(pose_to_matrix(pose) @ prior_matrix) for pose in poses]
            )
        source_mean[side] = _mean_pose(poses)
    if len(active) == 2:
        source_axis = source_mean["left"][:3] - source_mean["right"][:3]
        target_axis = target_mean["left"][:3] - target_mean["right"][:3]
        if np.linalg.norm(source_axis) < 1e-5:
            raise ValueError("Source bimanual end effectors have no usable separation")
        base_rotation = _align_vector_with_up(source_axis, target_axis)
    else:
        base_rotation = Rotation.identity()

    source_center = np.mean([source_mean[side][:3] for side in active], axis=0)
    target_center = np.mean([target_mean[side][:3] for side in active], axis=0)
    translation = target_center - base_rotation.apply(source_center)
    shared_base_pose = np.concatenate([translation, base_rotation.as_quat()])
    base_poses: dict[str, list[float]] = {side: shared_base_pose.tolist() for side in active}
    tool_poses: dict[str, list[float]] = {}
    for side in active:
        prior = tool_prior(side)
        if prior is not None:
            tool_poses[side] = prior.tolist()
        else:
            base_matrix = pose_to_matrix(shared_base_pose)
            mapped = base_matrix @ pose_to_matrix(source_mean[side])
            tool = np.linalg.inv(mapped) @ pose_to_matrix(target_mean[side])
            # Without CAD, fit only the axis convention; translation would pretend to know TCP.
            tool[:3, 3] = 0
            tool_poses[side] = matrix_to_pose(tool).tolist()

    # Inactive arms still need a complete serializable configuration.
    for side in SIDES:
        base_poses.setdefault(side, shared_base_pose.tolist())
        tool_poses.setdefault(side, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return {
        "method": "automatic_shared_frame_workspace_registration",
        "openarm_from_source_base": base_poses,
        "source_tool_from_openarm_tool": tool_poses,
        "position_scale": scale,
        "validated": False,
        "active_sides": list(active),
        "shared_base_frame": True,
        "tool_transform_method": (
            "source_config_cad_prior"
            if source_tool_from_openarm_tool is not None
            else "mean_orientation_fit_zero_translation"
        ),
        "warning": "Kinematic registration only; camera/scene metrology has not been validated",
    }
