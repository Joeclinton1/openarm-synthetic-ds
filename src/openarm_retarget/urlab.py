from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import ARM_JOINT_NAMES, OPENARM_MUJOCO_COMMIT, SIDES
from .model import resolve_model
from .schema import Episode


def _camera_frames(camera: dict, frame_count: int) -> list[np.ndarray]:
    frames = camera.get("world_from_camera_frames") or [camera.get("world_from_camera")]
    if not frames or frames[0] is None:
        raise ValueError("Camera JSON requires world_from_camera or world_from_camera_frames")
    if len(frames) not in (1, frame_count):
        raise ValueError("Camera trajectory length does not match the episode")
    if len(frames) == 1:
        frames = frames * frame_count
    result = [np.asarray(frame, dtype=np.float64) for frame in frames]
    if any(frame.shape != (4, 4) for frame in result):
        raise ValueError("Every camera transform must be 4x4")
    return result


def _opencv_camera_to_mujoco_pose(world_from_camera: np.ndarray) -> np.ndarray:
    """Convert OpenCV (+x right, +y down, +z forward) to MuJoCo camera axes."""
    world_from_mujoco = world_from_camera.copy()
    world_from_mujoco[:3, :3] = world_from_camera[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    quat = Rotation.from_matrix(world_from_mujoco[:3, :3]).as_quat()
    return np.concatenate([world_from_mujoco[:3, 3], quat])


def export_urlab_job(
    episode: Episode,
    destination: str | Path,
    camera_json: str | Path,
    model_path: str | Path | None = None,
    *,
    width: int = 960,
    height: int = 720,
    max_frames: int | None = None,
) -> Path:
    """Create a compact puppet-mode URLab replay job for high-throughput Unreal rendering."""
    if episode.joint_position is None:
        raise ValueError("URLab export requires an IK-solved episode")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model = resolve_model(model_path).resolve()
    camera_path = Path(camera_json).resolve()
    camera = json.loads(camera_path.read_text())
    frame_count = (
        len(episode.timestamp) if max_frames is None else min(max_frames, len(episode.timestamp))
    )
    if frame_count < 1:
        raise ValueError("URLab job must contain at least one frame")
    camera_frames = _camera_frames(camera, len(episode.timestamp))[:frame_count]
    camera_pose = np.stack([_opencv_camera_to_mujoco_pose(frame) for frame in camera_frames])
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("Camera intrinsics must be 3x3")

    # MuJoCo resolves compiler mesh directories from the top-level XML, not the included file.
    # Keep the wrapper beside a self-contained copy of the pinned model so URLab imports it on
    # local or remote render workers without rewriting asset paths.
    model_bundle = destination / "model"
    shutil.copytree(model.parent, model_bundle, dirs_exist_ok=True)
    bundled_model = model_bundle / model.name
    wrapper = model_bundle / "openarm_urlab.xml"
    wrapper.write_text(
        '<mujoco model="openarm_urlab_rerender">\n'
        f'  <include file="{bundled_model.name}"/>\n'
        "  <worldbody>\n"
        '    <body name="rerender_camera_rig" mocap="true">\n'
        '      <camera name="rerender_rgb" mode="fixed" '
        f'focalpixel="{intrinsics[0, 0]:.12g} {intrinsics[1, 1]:.12g}" '
        f'principalpixel="{intrinsics[0, 2]:.12g} {intrinsics[1, 2]:.12g}" '
        f'sensorsize="{width} {height}" resolution="{width} {height}"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    qpos = episode.joint_position[:frame_count].reshape(frame_count, -1)
    trajectory = destination / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        timestamp=episode.timestamp[:frame_count],
        arm_qpos=qpos,
        camera_pose_xyzw=camera_pose,
    )
    payload = {
        "schema": "openarm-urlab-puppet-job-v1",
        "model": str(wrapper),
        "source_model": str(model),
        "model_revision": OPENARM_MUJOCO_COMMIT,
        "trajectory": str(trajectory),
        "frames": frame_count,
        "fps": float(1.0 / episode.sample_period),
        "resolution": [width, height],
        "camera": "rerender_rgb",
        "camera_rig_body": "rerender_camera_rig",
        "camera_source": str(camera_path),
        "joint_order": [name for side in SIDES for name in ARM_JOINT_NAMES[side]],
        "registration_validated": bool(episode.metadata.get("calibrated", False)),
        "render_contract": {
            "mode": "puppet",
            "physics_steps_per_frame": 0,
            "transport": "shm",
            "pixel_format": "BGRA8",
            "alpha_required": True,
        },
    }
    job = destination / "urlab_job.json"
    job.write_text(json.dumps(payload, indent=2) + "\n")
    return job


def urlab_doctor(plugin_path: str | Path | None = None) -> dict:
    """Report the independently verifiable prerequisites for a URLab render worker."""
    candidates = [
        os.environ.get("UE_ROOT"),
        os.environ.get("UNREAL_ENGINE_ROOT"),
        "/opt/UnrealEngine",
        str(Path.home() / "UnrealEngine"),
    ]
    editor = None
    for candidate in filter(None, candidates):
        path = Path(candidate) / "Engine/Binaries/Linux/UnrealEditor"
        if path.exists():
            editor = str(path.resolve())
            break
    plugin = (
        Path(plugin_path).resolve() if plugin_path else Path("vendor/unreal_robotics_lab").resolve()
    )
    bridge_available = importlib.util.find_spec("urlab_client") is not None
    return {
        "ok": bool(editor and (plugin / "UnrealRoboticsLab.uplugin").exists() and bridge_available),
        "unreal_editor": editor,
        "required_unreal_version": "5.7",
        "plugin": str(plugin),
        "plugin_present": (plugin / "UnrealRoboticsLab.uplugin").exists(),
        "urlab_client_available": bridge_available,
        "ffmpeg": shutil.which("ffmpeg"),
        "gpu": shutil.which("nvidia-smi") is not None,
    }


def render_urlab_job(
    job_path: str | Path,
    output_dir: str | Path,
    *,
    address: str = "tcp://localhost",
    transport: str = "shm",
) -> Path:
    """Replay joint states in a running URLab 5.7 editor and capture camera frames."""
    try:
        from urlab_client import URLabAsset, URLabClient
    except ImportError as error:
        raise RuntimeError(
            "The URLab 5.7 Python client is not installed; follow the plugin's Python guide"
        ) from error
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    trajectory = np.load(job["trajectory"])
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with URLabClient(address, step_mode="puppet", transport=transport) as client:
        client.discover()
        handles = client.scene.apply_scene(
            "OpenArmRerender",
            [URLabAsset(actor_id="openarm", xml=job["model"], location=(0, 0, 0))],
            save=True,
        )
        client.sim.start(timeout_s=120.0)
        robot = handles["openarm"].runtime(client)
        model = client.model
        if model is None or client.data is None:
            raise RuntimeError("URLab puppet mode did not provide a local MuJoCo model")
        qpos_addresses = []
        for name in job["joint_order"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise KeyError(f"URLab model is missing joint {name}")
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        camera = robot.cameras.get(job["camera"]) or client.global_cameras[job["camera"]]
        for frame, (arm_qpos, camera_pose) in enumerate(
            zip(trajectory["arm_qpos"], trajectory["camera_pose_xyzw"], strict=True)
        ):
            client.data.qpos[qpos_addresses] = arm_qpos
            client.runtime.set_mocap_pose(
                job["camera_rig_body"],
                pos=tuple(camera_pose[:3]),
                quat=tuple(camera_pose[3:]),
            )
            client.step(n_steps=0, include_cameras=True)
            image = camera.latest_frame
            if image is None:
                raise RuntimeError(f"URLab returned no camera frame at index {frame}")
            if not cv2.imwrite(str(output_dir / f"{frame:06d}.png"), image):
                raise RuntimeError(f"Could not write URLab frame {frame}")
        client.sim.stop()
    elapsed = time.perf_counter() - started
    manifest = output_dir / "urlab_render_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "job": str(job_path),
                "frames": int(job["frames"]),
                "elapsed_seconds": elapsed,
                "frames_per_second": float(job["frames"] / elapsed),
                "transport": transport,
                "address": address,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest
