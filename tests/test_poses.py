import numpy as np
from scipy.spatial.transform import Rotation

from openarm_retarget.poses import (
    PoseTransform,
    compose_pose,
    invert_pose,
    matrix_to_pose,
    pose_to_matrix,
)


def test_pose_matrix_roundtrip() -> None:
    pose = np.r_[np.array([0.3, -0.2, 0.7]), Rotation.from_euler("xyz", [0.2, 0.4, -1]).as_quat()]
    np.testing.assert_allclose(
        pose_to_matrix(matrix_to_pose(pose_to_matrix(pose))), pose_to_matrix(pose)
    )


def test_inverse_composes_to_identity() -> None:
    pose = np.r_[np.array([1, 2, 3]), Rotation.from_euler("z", 0.5).as_quat()]
    identity = compose_pose(pose, invert_pose(pose))
    np.testing.assert_allclose(identity[:3], 0, atol=1e-12)
    np.testing.assert_allclose(np.abs(identity[6]), 1, atol=1e-12)


def test_pose_transform_scales_before_rigid_transform() -> None:
    transform = PoseTransform(
        np.array([1, 0, 0, 0, 0, 0, 1.0]), np.array([0, 0, 0, 0, 0, 0, 1.0]), 0.001
    )
    output = transform.apply(np.array([1000, 0, 0, 0, 0, 0, 1.0]))
    np.testing.assert_allclose(output[:3], [2, 0, 0])
