import numpy as np

from openarm_retarget.ik import OpenArmIK
from openarm_retarget.registration import auto_register_episode
from openarm_retarget.schema import Episode, SourceConfig


def test_auto_registration_uses_one_shared_openarm_base_frame(openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path)
    source = np.zeros((5, 2, 7))
    source[..., 6] = 1
    source[:, 0, :3] = [0.6, -0.3, 0.8]
    source[:, 1, :3] = [0.6, 0.3, 0.8]
    episode = Episode(
        timestamp=np.arange(5) / 30,
        ee_pose=source,
        gripper=np.zeros((5, 2)),
        task="test",
        source_dataset="test",
        source_episode="0",
        metadata={"active_sides": ["right", "left"]},
    )
    registration = auto_register_episode(episode, openarm_model_path)
    config = SourceConfig(
        name="test",
        repo_id="test",
        adapter="lerobot",
        position_scale=registration["position_scale"],
        openarm_from_source_base=registration["openarm_from_source_base"],
        source_tool_from_openarm_tool=registration["source_tool_from_openarm_tool"],
    )
    mapped_positions = []
    for index, side in enumerate(("right", "left")):
        mapped = config.pose_transform(side).apply(source[0, index])
        target = solver.forward_pose(side, solver.neutral(side))
        mapped_positions.append(mapped[:3])
        assert abs(float(np.dot(mapped[3:], target[3:]))) > 1 - 1e-6
    target_center = np.mean(
        [solver.forward_pose(side, solver.neutral(side))[:3] for side in ("right", "left")], axis=0
    )
    np.testing.assert_allclose(np.mean(mapped_positions, axis=0), target_center, atol=1e-6)
    np.testing.assert_allclose(
        registration["openarm_from_source_base"]["right"],
        registration["openarm_from_source_base"]["left"],
        atol=1e-12,
    )
    assert registration["shared_base_frame"] is True


def test_single_arm_registration_serializes_same_complete_base_for_inactive_arm(
    openarm_model_path,
) -> None:
    source = np.zeros((5, 2, 7))
    source[..., 6] = 1
    source[:, 0, :3] = [0.55, -0.2, 0.65]
    source[:, 0, 0] += np.linspace(0, 0.1, 5)
    episode = Episode(
        timestamp=np.arange(5) / 20,
        ee_pose=source,
        gripper=np.zeros((5, 2)),
        task="single arm",
        source_dataset="test",
        source_episode="0",
        metadata={"active_sides": ["right"]},
    )
    registration = auto_register_episode(episode, openarm_model_path)
    right = registration["openarm_from_source_base"]["right"]
    left = registration["openarm_from_source_base"]["left"]
    assert len(right) == len(left) == 7
    np.testing.assert_allclose(right, left, atol=1e-12)
    assert registration["source_tool_from_openarm_tool"]["left"] == [0, 0, 0, 0, 0, 0, 1]


def test_auto_registration_preserves_cad_tool_translation(openarm_model_path) -> None:
    source = np.zeros((5, 2, 7))
    source[..., 6] = 1
    source[:, 0, :3] = [0.55, -0.2, 0.65]
    episode = Episode(
        timestamp=np.arange(5) / 30,
        ee_pose=source,
        gripper=np.zeros((5, 2)),
        task="test",
        source_dataset="test",
        source_episode="0",
        metadata={"active_sides": ["right"]},
    )
    prior = {
        "right": [0.0, 0.0014, -0.06334, 0.70710678, 0.70710678, 0.0, 0.0],
        "left": [0, 0, 0, 0, 0, 0, 1],
    }
    registration = auto_register_episode(
        episode, openarm_model_path, source_tool_from_openarm_tool=prior
    )
    np.testing.assert_allclose(
        registration["source_tool_from_openarm_tool"]["right"], prior["right"]
    )
    assert registration["tool_transform_method"] == "source_config_cad_prior"


def test_auto_registration_preserves_source_base_rotation_prior(openarm_model_path) -> None:
    source = np.zeros((5, 2, 7))
    source[..., 6] = 1
    source[:, 0, :3] = [0.55, -0.2, 0.65]
    episode = Episode(
        timestamp=np.arange(5) / 30,
        ee_pose=source,
        gripper=np.zeros((5, 2)),
        task="test",
        source_dataset="test",
        source_episode="0",
        metadata={"active_sides": ["right"]},
    )
    prior = [0.12, -0.34, 0.56, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    registration = auto_register_episode(
        episode,
        openarm_model_path,
        openarm_from_source_base=prior,
        position_scale=0.73,
    )
    np.testing.assert_allclose(
        registration["openarm_from_source_base"]["right"][:3], prior[:3]
    )
    np.testing.assert_allclose(
        np.abs(registration["openarm_from_source_base"]["right"][3:]),
        np.abs(prior[3:]),
    )
    assert registration["position_scale"] == 0.73
