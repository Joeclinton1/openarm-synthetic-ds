import numpy as np

from openarm_retarget.ik import OpenArmIK
from openarm_retarget.schema import Episode


def test_inactive_arm_is_parked(openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path)
    target = solver.forward_pose("right", solver.neutral("right"))
    pose = np.zeros((2, 2, 7))
    pose[:, 0] = target
    pose[:, 1, 6] = 1
    episode = Episode(
        timestamp=np.arange(2) / 30,
        ee_pose=pose,
        gripper=np.zeros((2, 2)),
        task="single arm",
        source_dataset="test",
        source_episode="0",
        metadata={"active_sides": ["right"]},
    )
    solver.solve_episode(episode)
    np.testing.assert_allclose(
        episode.joint_position[:, 1], np.tile(solver.neutral("left"), (2, 1))
    )
    assert episode.diagnostics["ik_success"].all()
