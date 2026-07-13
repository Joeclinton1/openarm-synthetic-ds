import numpy as np

from openarm_retarget.ik import IKConfig, OpenArmIK


def test_ik_round_trip(openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path, IKConfig(max_iterations=200))
    for side in ("right", "left"):
        q = solver.neutral(side)
        q[2] += 0.2
        q[3] += 0.15
        pose = solver.forward_pose(side, q)
        result = solver.solve(side, pose, seed=solver.neutral(side))
        assert result.success
        assert result.position_error < solver.config.position_tolerance
        assert result.orientation_error < solver.config.orientation_tolerance


def test_unreachable_target_fails(openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path, IKConfig(max_iterations=20))
    result = solver.solve("right", np.array([10, 0, 0, 0, 0, 0, 1.0]))
    assert not result.success


def test_neutral_gripper_contact_is_not_invalid_collision(openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path)
    assert solver.arm_collisions(solver.neutral("right"), solver.neutral("left")) == []


def test_reference_posture_is_not_singular(openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path)
    for side in ("right", "left"):
        q = solver.neutral(side)
        pose = solver.forward_pose(side, q)
        assert solver.evaluate(side, q, pose)[2] < 20


def test_episode_recovers_after_unreachable_prefix(openarm_model_path) -> None:
    from openarm_retarget.schema import Episode

    solver = OpenArmIK(openarm_model_path)
    reachable = solver.forward_pose("right", solver.neutral("right"))
    poses = np.zeros((8, 2, 7))
    poses[..., 6] = 1
    poses[:, 0] = reachable
    poses[:2, 0, :3] = 10
    poses[:, 1] = solver.forward_pose("left", solver.neutral("left"))
    episode = Episode(
        timestamp=np.arange(8) / 30,
        ee_pose=poses,
        gripper=np.zeros((8, 2)),
        task="recovery",
        source_dataset="test",
        source_episode="0",
        metadata={"active_sides": ["right"]},
    )
    solver.solve_episode(episode)
    assert episode.diagnostics["ik_raw_success"][2:, 0].all()
