import numpy as np

from openarm_retarget.retarget import apply_registration
from openarm_retarget.schema import Episode, SourceConfig


def test_apply_registration_only_moves_active_arm() -> None:
    pose = np.zeros((2, 2, 7))
    pose[..., 6] = 1
    episode = Episode(
        timestamp=np.array([0.0, 0.1]),
        ee_pose=pose,
        gripper=np.zeros((2, 2)),
        task="test",
        source_dataset="test",
        source_episode="0",
        metadata={"active_sides": ["right"]},
    )
    registration = {
        "openarm_from_source_base": [1, 2, 3, 0, 0, 0, 1],
        "source_tool_from_openarm_tool": [0, 0, 0, 0, 0, 0, 1],
        "position_scale": 1,
        "validated": False,
    }
    result = apply_registration(
        episode, SourceConfig(name="test", repo_id="test", adapter="lerobot"), registration
    )
    np.testing.assert_allclose(result.ee_pose[:, 0, :3], np.tile([1, 2, 3], (2, 1)))
    np.testing.assert_allclose(result.ee_pose[:, 1, :3], 0)
