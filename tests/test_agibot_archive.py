import json

from openarm_retarget.agibot_archive import _archive_range, _episode_frames


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
