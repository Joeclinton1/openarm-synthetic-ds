import json

import mujoco
import numpy as np

from openarm_retarget.ik import OpenArmIK
from openarm_retarget.raytrace import (
    _frame_ranges,
    configure_blender_environment,
    export_blender_scene,
    render_blender_batch,
)
from openarm_retarget.schema import Episode
from openarm_retarget.viewer import TrajectoryViewer


def test_blender_scene_contains_official_meshes_and_motion(tmp_path, openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path)
    joints = np.zeros((2, 2, 7))
    for side_index, side in enumerate(("right", "left")):
        joints[:, side_index] = solver.neutral(side)
    joints[1, 0, 2] += 0.1
    episode = Episode(
        timestamp=np.array([0.0, 1 / 30]),
        ee_pose=np.zeros((2, 2, 7)),
        gripper=np.array([[0.0, 0.0], [1.0, 1.0]]),
        task="ray trace test",
        source_dataset="test",
        source_episode="0",
        joint_position=joints,
        metadata={"calibrated": False},
    )
    output = export_blender_scene(
        episode, tmp_path / "scene", openarm_model_path, width=320, height=240, samples=4
    )
    payload = json.loads(output.read_text())
    assert payload["engine"] == "CYCLES"
    assert payload["transparent_background"]
    assert payload["lighting"]["world_strength"] == 0.35
    assert payload["lighting"]["preset"] == "openarm-calibratable-studio-v2"
    assert payload["color_management"]["exposure"] == 0.0
    assert payload["episode_frames"] == 2
    assert len(payload["objects"]) > 20
    assert all((output.parent / item["mesh"]).is_file() for item in payload["objects"])
    right_link = next(item for item in payload["objects"] if item["name"] == "link3_right_00")
    assert right_link["material"]["color_space"] == "sRGB"
    assert right_link["world_from_object_frames"][0] != right_link["world_from_object_frames"][1]
    left_finger = next(
        item for item in payload["objects"] if item["name"] == "finger_inner_left_00"
    )
    assert left_finger["world_from_object_frames"][0] != left_finger["world_from_object_frames"][1]
    assert payload["gripper"]["semantics"] == "normalized 0=open, 1=closed"
    right_pinch = payload["anchors"]["right"]["pinch_center_world_frames"]
    assert len(right_pinch) == 2
    assert right_pinch[0] != right_pinch[1]
    viewer = TrajectoryViewer(openarm_model_path)
    geom_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_GEOM, "finger_inner_left_00")
    for frame in range(2):
        viewer.set_frame(episode, frame)
        transform = np.eye(4)
        transform[:3, :3] = viewer.data.geom_xmat[geom_id].reshape(3, 3)
        transform[:3, 3] = viewer.data.geom_xpos[geom_id]
        np.testing.assert_allclose(left_finger["world_from_object_frames"][frame], transform)


def test_blender_scene_supports_eevee(tmp_path, openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path)
    joints = np.array([[solver.neutral("right"), solver.neutral("left")]])
    episode = Episode(
        timestamp=np.array([0.0]),
        ee_pose=np.zeros((1, 2, 7)),
        gripper=np.zeros((1, 2)),
        task="eevee test",
        source_dataset="test",
        source_episode="0",
        joint_position=joints,
    )
    output = export_blender_scene(
        episode, tmp_path / "eevee", openarm_model_path, width=64, height=48, samples=0
    )
    payload = json.loads(output.read_text())
    assert payload["engine"] == "BLENDER_EEVEE_NEXT"
    assert payload["eevee_samples"] == 16


def test_blender_environment_configuration_preserves_scene(tmp_path, openarm_model_path) -> None:
    solver = OpenArmIK(openarm_model_path)
    joints = np.array([[solver.neutral("right"), solver.neutral("left")]])
    episode = Episode(
        timestamp=np.array([0.0]),
        ee_pose=np.zeros((1, 2, 7)),
        gripper=np.zeros((1, 2)),
        task="hdri test",
        source_dataset="test",
        source_episode="0",
        joint_position=joints,
    )
    scene = export_blender_scene(episode, tmp_path / "scene", openarm_model_path)
    environment = tmp_path / "probe.exr"
    environment.write_bytes(b"test")
    output = configure_blender_environment(
        scene, scene.parent / "scene_hdri.json", environment, strength=0.8, area_light_scale=0.2
    )
    original = json.loads(scene.read_text())
    configured = json.loads(output.read_text())
    assert configured["objects"] == original["objects"]
    assert configured["camera"] == original["camera"]
    assert configured["lighting"]["environment_path"] == str(environment.resolve())
    assert configured["lighting"]["world_strength"] == 0.8
    assert configured["lighting"]["area_lights"][0]["energy_w"] == 32.0


def test_frame_ranges_cover_every_frame_once() -> None:
    ranges = _frame_ranges(10, 3)
    assert ranges == [(0, 4), (4, 7), (7, 10)]
    assert [frame for start, end in ranges for frame in range(start, end)] == list(range(10))


def test_completed_blender_batch_resume_does_not_launch_blender(tmp_path) -> None:
    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "engine": "BLENDER_EEVEE_NEXT",
                "episode_frames": 2,
                "resolution": [64, 48],
                "samples": 0,
            }
        )
    )
    output = tmp_path / "rgba"
    output.mkdir()
    for frame in range(2):
        (output / f"{frame:06d}.png").write_bytes(b"x" * 101)
    manifest = render_blender_batch(scene, output, blender=tmp_path / "does-not-exist", resume=True)
    report = json.loads(manifest.read_text())
    assert report["resume_noop"]
    assert report["new_frames"] == 0
    assert report["resumed_frames"] == 2
