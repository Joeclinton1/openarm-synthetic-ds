import json

import numpy as np
import pyarrow.parquet as pq

from openarm_retarget.export import export_lerobot_v3, validate_lerobot_v3
from openarm_retarget.schema import Episode


def test_export_openarm_order_and_gripper_sign(tmp_path) -> None:
    n = 4
    pose = np.zeros((n, 2, 7))
    pose[..., 6] = 1
    episode = Episode(
        timestamp=np.arange(n) / 30,
        ee_pose=pose,
        gripper=np.tile([1.0, 0.5], (n, 1)),
        joint_position=np.stack(
            [
                np.tile(np.arange(7), (n, 1)),
                np.tile(np.arange(10, 17), (n, 1)),
            ],
            axis=1,
        ),
        feasible=np.array([True, True, False, True]),
        task="move",
        source_dataset="source",
        source_episode="1",
        metadata={"calibrated": True},
    )
    export_lerobot_v3([episode], tmp_path)
    table = pq.read_table(tmp_path / "data/chunk-000/file-000.parquet")
    assert table.num_rows == 3
    first = np.asarray(table["action"][0].as_py())
    np.testing.assert_array_equal(first[:7], np.arange(7))
    assert np.isclose(first[7], 0.0)
    np.testing.assert_array_equal(first[8:15], np.arange(10, 17))
    assert np.isclose(first[15], 0.3927)
    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["robot_type"] == "openarm_bimanual_v2.0"
    assert (tmp_path / "meta/stats.json").is_file()
    assert validate_lerobot_v3(tmp_path)["ok"]


def test_export_refuses_uncalibrated(tmp_path) -> None:
    import pytest

    pose = np.zeros((1, 2, 7))
    pose[..., 6] = 1
    episode = Episode(
        timestamp=np.array([0.0]),
        ee_pose=pose,
        gripper=np.zeros((1, 2)),
        joint_position=np.zeros((1, 2, 7)),
        feasible=np.ones(1, dtype=bool),
        task="move",
        source_dataset="source",
        source_episode="uncalibrated",
    )
    with pytest.raises(ValueError, match="Refusing to export uncalibrated"):
        export_lerobot_v3([episode], tmp_path)
