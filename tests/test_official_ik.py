import numpy as np
import pytest

from openarm_retarget.ik import OpenArmIK
from openarm_retarget.official_ik import OfficialIKConfig, OfficialOpenArmIK
from openarm_retarget.schema import Episode


def _official_solver(model, iterations=80):
    pytest.importorskip("openarm_control")
    return OfficialOpenArmIK(model, OfficialIKConfig(max_iterations=iterations))


def test_official_ik_round_trip_bimanual(openarm_model_path) -> None:
    reference = OpenArmIK(openarm_model_path)
    solver = _official_solver(openarm_model_path)
    poses = np.zeros((2, 2, 7))
    poses[..., 6] = 1
    for side_index, side in enumerate(("right", "left")):
        q = reference.neutral(side)
        q[2] += 0.1 if side == "right" else -0.1
        poses[:, side_index] = reference.forward_pose(side, q)
    episode = Episode(
        timestamp=np.array([0.0, 1 / 30]),
        ee_pose=poses,
        gripper=np.zeros((2, 2)),
        task="official IK round trip",
        source_dataset="test",
        source_episode="0",
    )
    solver.solve_episode(episode)
    assert episode.diagnostics["official_solver_failed"].sum() == 0
    assert episode.diagnostics["ik_success"][-1].all()
    assert episode.metadata["ik_solver"]["implementation"]["release"] == "0.1.0"


def test_official_pose_converts_xyzw_to_wxyz() -> None:
    converted = OfficialOpenArmIK._official_pose(np.array([1, 2, 3, 0.1, 0.2, 0.3, 0.9]))
    np.testing.assert_allclose(converted, [1, 2, 3, 0.9, 0.1, 0.2, 0.3])
