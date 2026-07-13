import h5py
import numpy as np

from openarm_retarget.adapters.agibot import load_agibot_h5
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
    np.testing.assert_allclose(episode.gripper[0], 0, atol=0.02)
    np.testing.assert_allclose(episode.gripper[1], 1, atol=0.02)


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
