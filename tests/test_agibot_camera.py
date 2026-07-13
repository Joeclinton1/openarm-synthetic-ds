import json

import numpy as np

from openarm_retarget.camera import write_agibot_openarm_camera
from openarm_retarget.schema import Episode


def test_agibot_camera_uses_shared_similarity_transform(tmp_path) -> None:
    episode = Episode(
        timestamp=np.array([0.0]),
        ee_pose=np.array([[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]]),
        gripper=np.zeros((1, 2)),
        task="test",
        source_dataset="test",
        source_episode="0",
        metadata={
            "registration": {
                "shared_base_frame": True,
                "validated": False,
                "position_scale": 0.5,
                "openarm_from_source_base": {
                    "right": [1, 2, 3, 0, 0, 0, 1],
                    "left": [1, 2, 3, 0, 0, 0, 1],
                },
            }
        },
    )
    episode_path = tmp_path / "episode.npz"
    episode.save(episode_path)
    intrinsic_path = tmp_path / "intrinsic.json"
    intrinsic_path.write_text(
        json.dumps(
            {
                "intrinsic": {
                    "fx": 100,
                    "fy": 101,
                    "ppx": 50,
                    "ppy": 40,
                    "distortion_model": "plumb bob",
                }
            }
        )
    )
    extrinsic_path = tmp_path / "extrinsic.json"
    extrinsic_path.write_text(
        json.dumps(
            [
                {
                    "extrinsic": {
                        "rotation_matrix": np.eye(3).tolist(),
                        "translation_vector": [2, 4, 6],
                    }
                }
            ]
        )
    )
    output = write_agibot_openarm_camera(
        episode_path, intrinsic_path, extrinsic_path, tmp_path / "camera.json"
    )
    camera = json.loads(output.read_text())
    np.testing.assert_allclose(np.asarray(camera["world_from_camera_frames"])[0, :3, 3], [2, 4, 6])
    assert camera["intrinsics"][1][1] == 101
