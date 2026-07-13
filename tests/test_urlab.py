import json

import numpy as np
import pytest

from openarm_retarget.urlab import (
    _opencv_camera_to_mujoco_pose,
    export_urlab_job,
    plan_urlab_shards,
    urlab_doctor,
    validate_urlab_job,
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
    assert job["schema"] == "openarm-urlab-job-v2"
    assert job["frames"] == 3
    assert job["render_contract"]["transport"] == "shm"
    trajectory = np.load(job["trajectory"])
    assert trajectory["arm_qpos"].shape == (3, 14)
    assert trajectory["finger_qpos"].shape == (3, 4)
    assert trajectory["world_from_camera"].shape == (3, 4, 4)
    assert not (tmp_path / "job/model/assets").exists()
    assert validate_urlab_job(job_path)["ok"]
    wrapper = (tmp_path / "job/model/openarm_urlab.xml").read_text()
    assert 'focalpixel="400 410"' in wrapper
    assert 'resolution="640 480"' in wrapper


def test_urlab_doctor_is_explicit_about_missing_runtime(tmp_path) -> None:
    report = urlab_doctor(tmp_path)
    assert report["required_unreal_version"] == "5.7"
    assert not report["plugin_present"]


def test_urlab_grippers_expand_with_official_signs(openarm_model_path, tmp_path) -> None:
    poses = np.zeros((2, 2, 7))
    poses[..., 6] = 1
    episode = Episode(
        timestamp=np.arange(2) / 30,
        ee_pose=poses,
        gripper=np.array([[0.0, 0.0], [1.0, 0.5]]),
        task="test",
        source_dataset="test",
        source_episode="0",
        joint_position=np.zeros((2, 2, 7)),
    )
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "intrinsics": [[400, 0, 300], [0, 410, 200], [0, 0, 1]],
                "world_from_camera": np.eye(4).tolist(),
            }
        )
    )
    job_path = export_urlab_job(episode, tmp_path / "job", camera, openarm_model_path)
    job = json.loads(job_path.read_text())
    with np.load(job["trajectory"], allow_pickle=False) as trajectory:
        np.testing.assert_allclose(trajectory["finger_qpos"][1], [0.0, 0.0, 0.3927, 0.3927])


def test_urlab_job_rejects_tampered_trajectory(openarm_model_path, tmp_path) -> None:
    poses = np.zeros((1, 2, 7))
    poses[..., 6] = 1
    episode = Episode(
        timestamp=np.array([0.0]),
        ee_pose=poses,
        gripper=np.zeros((1, 2)),
        task="test",
        source_dataset="test",
        source_episode="0",
        joint_position=np.zeros((1, 2, 7)),
    )
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "intrinsics": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
                "world_from_camera": np.eye(4).tolist(),
            }
        )
    )
    job_path = export_urlab_job(episode, tmp_path / "job", camera, openarm_model_path)
    (job_path.parent / "trajectory.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_urlab_job(job_path)


def test_urlab_shards_have_stable_warmup_overlap() -> None:
    assert plan_urlab_shards(10, 4, 2) == [
        {"warmup_start": 0, "retain_start": 0, "retain_end": 4},
        {"warmup_start": 2, "retain_start": 4, "retain_end": 8},
        {"warmup_start": 6, "retain_start": 8, "retain_end": 10},
    ]
