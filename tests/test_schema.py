import numpy as np

from openarm_retarget.schema import Episode, SourceConfig


def make_episode() -> Episode:
    pose = np.zeros((3, 2, 7))
    pose[..., 6] = 1
    return Episode(
        timestamp=np.array([0.0, 0.1, 0.2]),
        ee_pose=pose,
        gripper=np.zeros((3, 2)),
        task="test",
        source_dataset="test/source",
        source_episode="0",
    )


def test_episode_roundtrip(tmp_path) -> None:
    episode = make_episode()
    episode.gripper_width_m = np.full((3, 2), 0.04)
    path = tmp_path / "episode.npz"
    episode.save(path)
    loaded = Episode.load(path)
    assert loaded.task == "test"
    np.testing.assert_array_equal(loaded.timestamp, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(loaded.gripper_width_m, 0.04)


def test_rejects_nonmonotonic_timestamp() -> None:
    episode = make_episode()
    episode.timestamp[2] = 0.05
    try:
        episode.validate()
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("invalid timestamp accepted")


def test_slice_resets_time() -> None:
    episode = make_episode()
    episode.gripper_width_m = np.full((3, 2), 0.04)
    sliced = episode.sliced(1, 3)
    np.testing.assert_allclose(sliced.timestamp, [0, 0.1])
    assert sliced.metadata["source_frame_slice"] == [1, 3]
    assert sliced.gripper_width_m.shape == (2, 2)


def test_source_config_selects_per_side_tool_transform() -> None:
    config = SourceConfig(
        name="source",
        repo_id="example/source",
        adapter="lerobot",
        source_tool_from_openarm_tool={
            "right": [1, 0, 0, 0, 0, 0, 1],
            "left": [-1, 0, 0, 0, 0, 0, 1],
        },
    )
    np.testing.assert_allclose(
        config.pose_transform("right").source_tool_from_openarm_tool[:3], [1, 0, 0]
    )
    np.testing.assert_allclose(
        config.pose_transform("left").source_tool_from_openarm_tool[:3], [-1, 0, 0]
    )
