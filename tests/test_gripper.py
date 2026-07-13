import numpy as np

from openarm_retarget.gripper import (
    OPENARM_CLOSED_PINCH_MIDPOINT_M,
    OPENARM_FINGER_MAX_ANGLE_RAD,
    OPENARM_MAX_APERTURE_M,
    OpenArmPinchKinematics,
    aperture_to_closure,
    closure_to_aperture,
    closure_to_finger_qpos,
    opening_angle_to_aperture,
    pinch_midpoint_local,
    preserve_pinch_center,
)
from openarm_retarget.gripper_validation import validate_gripper_contact
from openarm_retarget.ik import OpenArmIK
from openarm_retarget.poses import pose_to_matrix
from openarm_retarget.schema import Episode


def test_canonical_closure_maps_to_official_joint_limits() -> None:
    np.testing.assert_allclose(
        closure_to_finger_qpos(np.array([[0.0, 0.0], [1.0, 1.0]])),
        [
            [-OPENARM_FINGER_MAX_ANGLE_RAD] * 2 + [OPENARM_FINGER_MAX_ANGLE_RAD] * 2,
            [0.0] * 4,
        ],
    )


def test_physical_aperture_round_trip_is_monotonic() -> None:
    angles = np.linspace(0, OPENARM_FINGER_MAX_ANGLE_RAD, 101)
    apertures = opening_angle_to_aperture(angles)
    assert np.all(np.diff(apertures) > 0)
    closures = aperture_to_closure(apertures)
    np.testing.assert_allclose(closure_to_aperture(closures), apertures, atol=1e-12)
    assert np.isclose(apertures[0], 0)
    assert np.isclose(apertures[-1], OPENARM_MAX_APERTURE_M)


def test_pinch_compensation_keeps_closed_contact_target_fixed() -> None:
    poses = np.zeros((5, 2, 7))
    poses[..., :3] = [0.3, -0.2, 0.5]
    poses[..., 6] = 1
    closure = np.tile(np.linspace(0, 1, 5)[:, None], (1, 2))
    compensated, target = preserve_pinch_center(poses, closure)
    transforms = pose_to_matrix(compensated)
    achieved = transforms[..., :3, 3] + np.einsum(
        "...ij,...j->...i", transforms[..., :3, :3], pinch_midpoint_local(closure)
    )
    np.testing.assert_allclose(achieved, target, atol=1e-12)
    np.testing.assert_allclose(
        target - poses[..., :3],
        np.broadcast_to(OPENARM_CLOSED_PINCH_MIDPOINT_M, target.shape),
    )


def test_mujoco_pad_geometry_matches_analytic_aperture(openarm_model_path) -> None:
    kinematics = OpenArmPinchKinematics(openarm_model_path)
    joints = np.zeros((2, 7))
    for closure in np.linspace(0, 1, 9):
        kinematics.set_frame(joints, np.array([closure, closure]))
        for side in ("right", "left"):
            _, aperture = kinematics.measure(side)
            assert np.isclose(aperture, closure_to_aperture(closure), atol=1e-12)


def test_contact_validator_measures_rendered_frame_state(openarm_model_path) -> None:
    ik = OpenArmIK(openarm_model_path)
    frames = 5
    joints = np.empty((frames, 2, 7))
    poses = np.empty((frames, 2, 7))
    for side_index, side in enumerate(("right", "left")):
        joints[:, side_index] = ik.neutral(side)
        poses[:, side_index] = ik.forward_pose(side, ik.neutral(side))
    episode = Episode(
        timestamp=np.arange(frames) / 30,
        ee_pose=poses,
        gripper=np.tile(np.linspace(0, 1, frames)[:, None], (1, 2)),
        task="contact validation",
        source_dataset="test",
        source_episode="0",
        joint_position=joints,
        diagnostics={"ik_success": np.ones((frames, 2), dtype=bool)},
    )
    report = validate_gripper_contact(episode, openarm_model_path)
    assert report["ok"]
    assert report["validated_frame_sides"] == frames * 2
    assert report["aperture_error_m"]["max"] < 1e-12
    assert report["pinch_error_m"]["max"] < 1e-12
