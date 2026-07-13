from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12) or not np.all(np.isfinite(norm)):
        raise ValueError("Quaternion contains a zero, non-finite, or invalid value")
    q = q / norm
    # q and -q are equivalent. A stable hemisphere prevents artificial discontinuities.
    return np.where(q[..., 3:4] < 0, -q, q)


def make_quaternions_continuous(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_xyzw(quaternion).copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    return q


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected (..., 7) pose, got {pose.shape}")
    result = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.float64)
    result[..., :3, :3] = Rotation.from_quat(normalize_quaternion_xyzw(pose[..., 3:])).as_matrix()
    result[..., :3, 3] = pose[..., :3]
    result[..., 3, 3] = 1.0
    return result


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (..., 4, 4) matrix, got {matrix.shape}")
    return np.concatenate(
        [matrix[..., :3, 3], Rotation.from_matrix(matrix[..., :3, :3]).as_quat()], axis=-1
    )


def compose_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pose A<-C from A<-B and B<-C."""
    return matrix_to_pose(pose_to_matrix(a) @ pose_to_matrix(b))


def invert_pose(pose: np.ndarray) -> np.ndarray:
    return matrix_to_pose(np.linalg.inv(pose_to_matrix(pose)))


def convert_quaternion_order(quaternion: np.ndarray, order: str) -> np.ndarray:
    q = np.asarray(quaternion)
    if order == "xyzw":
        return q
    if order == "wxyz":
        return q[..., [1, 2, 3, 0]]
    raise ValueError(f"Unsupported quaternion order: {order}")


@dataclass(frozen=True)
class PoseTransform:
    """Calibrated mapping from a source base/tool pose to an OpenArm base/tool pose."""

    openarm_from_source_base: np.ndarray
    source_tool_from_openarm_tool: np.ndarray
    position_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("openarm_from_source_base", "source_tool_from_openarm_tool"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (7,):
                raise ValueError(f"{name} must be a 7-vector")

    @classmethod
    def identity(cls) -> "PoseTransform":
        identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        return cls(identity.copy(), identity.copy())

    def apply(self, source_pose: np.ndarray) -> np.ndarray:
        source_pose = np.asarray(source_pose, dtype=np.float64).copy()
        source_pose[..., :3] *= self.position_scale
        return compose_pose(
            compose_pose(self.openarm_from_source_base, source_pose),
            self.source_tool_from_openarm_tool,
        )


def orientation_error(target_xyzw: np.ndarray, current_xyzw: np.ndarray) -> np.ndarray:
    target = Rotation.from_quat(normalize_quaternion_xyzw(target_xyzw))
    current = Rotation.from_quat(normalize_quaternion_xyzw(current_xyzw))
    return (target * current.inv()).as_rotvec()
