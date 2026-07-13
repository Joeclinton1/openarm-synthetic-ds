import json

import numpy as np

from openarm_retarget.urlab import (
    _opencv_camera_to_mujoco_pose,
    export_urlab_job,
    urlab_doctor,
)
from openarm_retarget.schema import Episode


def test_opencv_camera_axes_convert_to_mujoco() -> None:
    pose = _opencv_camera_to_mujoco_pose(np.eye(4))
    matrix = np.diag([1.0, -1.0, -1.0])
    from scipy.spatial.transform import Rotation

    np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), matrix, atol=1e-8)


def test_export_urlab_job(openarm_model_path, tmp_path) -> None:
    poses = np.zeros((4, 2, 7))
    poses[..., 6] = 1
    episode = Episode(
        timestamp=np.arange(4) / 30,
        ee_pose=poses,
        gripper=np.zeros((4, 2)),
        task="test",
        source_dataset="test",
        source_episode="0",
        joint_position=np.zeros((4, 2, 7)),
    )
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "intrinsics": [[400, 0, 320], [0, 410, 240], [0, 0, 1]],
                "world_from_camera": np.eye(4).tolist(),
            }
        )
    )
    job_path = export_urlab_job(
        episode,
        tmp_path / "job",
        camera,
        openarm_model_path,
        width=640,
        height=480,
        max_frames=3,
    )
    job = json.loads(job_path.read_text())
    assert job["frames"] == 3
    assert job["render_contract"]["transport"] == "shm"
    trajectory = np.load(job["trajectory"])
    assert trajectory["arm_qpos"].shape == (3, 14)
    wrapper = (tmp_path / "job/model/openarm_urlab.xml").read_text()
    assert 'focalpixel="400 410"' in wrapper
    assert 'resolution="640 480"' in wrapper


def test_urlab_doctor_is_explicit_about_missing_runtime(tmp_path) -> None:
    report = urlab_doctor(tmp_path)
    assert report["required_unreal_version"] == "5.7"
    assert not report["plugin_present"]
