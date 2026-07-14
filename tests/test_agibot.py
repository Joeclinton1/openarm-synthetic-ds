import h5py
import numpy as np

from openarm_retarget.adapters.agibot import load_agibot_h5
from openarm_retarget.gripper import aperture_to_closure, closure_to_aperture
from openarm_retarget.schema import SourceConfig


def test_agibot_width_gripper_is_inverted(tmp_path) -> None:
    path = tmp_path / "proprio_stats.h5"
    with h5py.File(path, "w") as data:
        data["timestamp"] = np.array([0, 1_000_000_000], dtype=np.int64)
        data["state/end/position"] = np.zeros((2, 2, 3))
        orientation = np.zeros((2, 2, 4))
        orientation[..., 3] = 1
        data["state/end/orientation"] = orientation
        data["state/effector/position"] = np.array([[100, 100], [20, 20]])
    config = SourceConfig(
        name="AgiBot",
        repo_id="agibot/test",
        adapter="agibot_h5",
        gripper_mode="width_mm",
    )
    episode = load_agibot_h5(path, config, allow_uncalibrated=True)
    np.testing.assert_allclose(episode.gripper[0], aperture_to_closure(0.1))
    np.testing.assert_allclose(episode.gripper[1], aperture_to_closure(0.02))
    np.testing.assert_allclose(episode.gripper_width_m, [[0.1, 0.1], [0.02, 0.02]])


def test_agibot_epoch_nanoseconds_keep_relative_precision(tmp_path) -> None:
    path = tmp_path / "proprio_stats.h5"
    base = 1_734_693_131_578_195_000
    with h5py.File(path, "w") as data:
        data["timestamp"] = np.array([base, base + 33_333_333, base + 66_666_667])
        data["state/end/position"] = np.zeros((3, 2, 3))
        orientation = np.zeros((3, 2, 4))
        orientation[..., 3] = 1
        data["state/end/orientation"] = orientation
        data["state/effector/position"] = np.ones((3, 2))
    config = SourceConfig(
        name="AgiBot",
        repo_id="agibot/test",
        adapter="agibot_h5",
        gripper_mode="width_mm",
    )
    episode = load_agibot_h5(path, config, allow_uncalibrated=True)
    np.testing.assert_allclose(episode.timestamp, [0, 0.033333333, 0.066666667], atol=1e-12)


def test_agibot_closure_position_increases_from_open_to_closed(tmp_path) -> None:
    path = tmp_path / "proprio_stats.h5"
    with h5py.File(path, "w") as data:
        data["timestamp"] = np.array([0, 1_000_000_000], dtype=np.int64)
        data["state/end/position"] = np.zeros((2, 2, 3))
        orientation = np.zeros((2, 2, 4))
        orientation[..., 3] = 1
        data["state/end/orientation"] = orientation
        data["state/effector/position"] = np.array([[34.7, 34.7], [117.3, 117.3]])
    config = SourceConfig(
        name="AgiBot",
        repo_id="agibot/test",
        adapter="agibot_h5",
        gripper_mode="closure_position",
        gripper_open_value=34.7,
        gripper_closed_value=117.3,
    )
    episode = load_agibot_h5(path, config, allow_uncalibrated=True)
    np.testing.assert_allclose(episode.gripper, [[0, 0], [1, 1]])
    np.testing.assert_allclose(episode.gripper_width_m, closure_to_aperture(episode.gripper))


def test_agibot_reconstructs_held_state_events_without_using_targets(tmp_path) -> None:
    path = tmp_path / "proprio_stats.h5"
    frames = 12
    with h5py.File(path, "w") as data:
        data["timestamp"] = np.arange(frames, dtype=np.int64) * 100_000_000
        data["state/end/position"] = np.zeros((frames, 2, 3))
        orientation = np.zeros((frames, 2, 4))
        orientation[..., 3] = 1
        data["state/end/orientation"] = orientation
        # Two sparse measured updates followed by a long stable run.
        held = np.array([0, 0, 0, 0.25, 0.25, 0.25, 1, 1, 1, 1, 1, 1])
        data["state/effector/position"] = np.tile(held[:, None], (1, 2))
        # Deliberately binary and earlier: interpolation must not use this target channel.
        target = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        data["action/effector/position"] = np.tile(target[:, None], (1, 2))
    config = SourceConfig(
        name="AgiBot",
        repo_id="agibot/test",
        adapter="agibot_h5",
        gripper_mode="closure_position",
        gripper_open_value=0,
        gripper_closed_value=1,
        gripper_interpolation="held_state_smoothstep",
        gripper_interpolation_max_gap_s=1.0,
        gripper_transition_duration_s=0.3,
    )

    episode = load_agibot_h5(path, config, allow_uncalibrated=True)

    np.testing.assert_allclose(episode.gripper[[0, 2, 5, 6, 11], 0], [0, 0, 1, 1, 1])
    assert 0 < episode.gripper[3, 0] < episode.gripper[4, 0] < 1
    np.testing.assert_allclose(
        episode.diagnostics["gripper_raw_normalized"][:, 0], held
    )
    assert episode.metadata["gripper_interpolation"] == "held_state_smoothstep"
