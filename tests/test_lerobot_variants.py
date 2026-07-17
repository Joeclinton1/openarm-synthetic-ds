import numpy as np
import pyarrow as pa

from openarm_retarget.adapters.lerobot import load_lerobot_episode
from openarm_retarget.schema import SourceConfig


def test_loads_single_arm_quaternion_and_calibrated_width():
    table = pa.table(
        {
            "episode_index": [0, 0],
            "timestamp": [0.0, 0.1],
            "pose": [[0.4, 0.0, 0.3, 0, 0, 0, 1]] * 2,
            "width": [80.8, 0.0],
        }
    )
    config = SourceConfig(
        name="RH20T Franka cfg5",
        repo_id="example/rh20t",
        adapter="lerobot",
        rotation_representation="quaternion",
        single_arm_side="right",
        gripper_mode="width_mm",
        gripper_open_value=80.8,
        gripper_closed_value=0.0,
        fields={"pose": "pose", "gripper": "width"},
    )
    episode = load_lerobot_episode(table, config, 0, allow_uncalibrated=True)
    np.testing.assert_allclose(episode.ee_pose[:, 0, 3:], [[0, 0, 0, 1]] * 2)
    np.testing.assert_allclose(episode.gripper[:, 0], [0, 1])
    assert episode.metadata["active_sides"] == ["right"]


def test_loads_bimanual_separate_euler_fields_with_trailing_gripper():
    table = pa.table(
        {
            "episode_index": [0, 0, 1],
            "timestamp": [0.0, 1 / 30, 0.0],
            "left_pose": [[0.1, 0.2, 0.3, 0, 0, 0, 99]] * 3,
            "right_pose": [[0.4, 0.5, 0.6, 0, 0, 0, 99]] * 3,
            "left_gripper": [0.0, 1.0, 0.5],
            "right_gripper": [1.0, 0.0, 0.5],
        }
    )
    config = SourceConfig(
        name="RoboMIND AgileX 3RGB",
        repo_id="example/robomind",
        adapter="lerobot",
        rotation_representation="euler",
        arm_order=["left", "right"],
        gripper_mode="signed",
        fields={
            "pose_left": "left_pose",
            "pose_right": "right_pose",
            "gripper_left": "left_gripper",
            "gripper_right": "right_gripper",
        },
    )
    episode = load_lerobot_episode(table, config, 0, allow_uncalibrated=True)
    np.testing.assert_allclose(episode.ee_pose[:, 1, :3], [[0.1, 0.2, 0.3]] * 2)
    np.testing.assert_allclose(episode.ee_pose[:, 0, :3], [[0.4, 0.5, 0.6]] * 2)
    np.testing.assert_allclose(
        episode.ee_pose[:, :, 3:], [[[0, 0, 0, 1], [0, 0, 0, 1]]] * 2
    )
    np.testing.assert_allclose(episode.gripper, [[1, 0], [0, 1]])
