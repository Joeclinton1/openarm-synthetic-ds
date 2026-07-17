import json

import numpy as np

from openarm_retarget.agibot_archive import (
    _archive_range,
    _episode_frames,
    _motion_active_sides,
)
from openarm_retarget.schema import Episode


def test_motion_active_sides_excludes_stationary_arm() -> None:
    pose = np.zeros((20, 2, 7))
    pose[..., 6] = 1
    pose[:, 1, 0] = np.linspace(0, 0.2, len(pose))
    episode = Episode(
        timestamp=np.arange(20) / 30,
        ee_pose=pose,
        gripper=np.zeros((20, 2)),
        task="test",
        source_dataset="agibot",
        source_episode="0",
    )
    assert _motion_active_sides(episode) == ["left"]


def test_archive_range() -> None:
    assert _archive_range("observations/410/648773-660695.tar") == (648773, 660695)
    assert _archive_range("sample_dataset.tar") is None


def test_episode_frames() -> None:
    annotation = {
        "label_info": {
            "action_config": [
                {"start_frame": 0, "end_frame": 10},
                {"start_frame": 10, "end_frame": 25},
            ]
        }
    }
    assert _episode_frames(json.loads(json.dumps(annotation))) == 25
