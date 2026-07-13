from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import (
    ARM_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    OPENARM_MUJOCO_COMMIT,
    SIDES,
    URLAB_BRIDGE_COMMIT,
    URLAB_COMMIT,
    URLAB_MODEL_ID,
)
from .gripper import aperture_to_closure, closure_to_finger_qpos
from .model import resolve_model
from .schema import Episode

JOB_SCHEMA = "openarm-urlab-job-v2"
CAMERA_NAMES = {
    "rgba": "openarm_rgb",
    "depth_m": "openarm_depth",
    "instance_segmentation": "openarm_instance",
    "shadow": "openarm_shadow",
}
DEFAULT_PASSES = ("rgba", "depth_m", "instance_segmentation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


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


def _validate_rigid_transforms(transforms: np.ndarray) -> None:
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("world_from_camera must have shape [frames,4,4]")
    if not np.all(np.isfinite(transforms)):
        raise ValueError("Camera transforms contain non-finite values")
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), transforms[:, 3].shape)
    if not np.allclose(transforms[:, 3], expected_bottom, atol=1e-8):
        raise ValueError("Camera transforms must have homogeneous bottom row [0,0,0,1]")
    rotations = transforms[:, :3, :3]
    identities = np.einsum("tji,tjk->tik", rotations, rotations)
    if not np.allclose(identities, np.eye(3), atol=1e-5):
        raise ValueError("Camera rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise ValueError("Camera rotations must have determinant +1")


def _opencv_camera_to_mujoco_pose(world_from_camera: np.ndarray) -> np.ndarray:
    """Convert OpenCV (+x right, +y down, +z forward) to MuJoCo camera axes."""
    world_from_mujoco = world_from_camera.copy()
    world_from_mujoco[:3, :3] = world_from_camera[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    quat = Rotation.from_matrix(world_from_mujoco[:3, :3]).as_quat()
    return np.concatenate([world_from_mujoco[:3, 3], quat])


def _validate_intrinsics(intrinsics: np.ndarray) -> None:
    if intrinsics.shape != (3, 3):
        raise ValueError("Camera intrinsics must be 3x3")
    if not np.all(np.isfinite(intrinsics)):
        raise ValueError("Camera intrinsics contain non-finite values")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-9):
        raise ValueError("Camera intrinsics bottom row must be [0,0,1]")


def _gripper_qpos(gripper: np.ndarray) -> np.ndarray:
    """Expand normalized right/left closure to both coupled finger joints."""
    return closure_to_finger_qpos(gripper)


def export_urlab_job(
    episode: Episode,
    destination: str | Path,
    camera_json: str | Path,
    model_path: str | Path | None = None,
    *,
    width: int = 960,
    height: int = 720,
    max_frames: int | None = None,
    lighting_preset: str = "openarm-neutral-studio-v1",
    material_preset: str = "openarm-official-srgb-v1",
    required_passes: tuple[str, ...] = DEFAULT_PASSES,
    hardware_ray_tracing: bool = False,
    lumen_quality: int = 2,
    seed: int = 0,
) -> Path:
    """Export a portable, validated URLab v2 replay job.

    Unlike v1, this never copies the 91 MB OpenArm model into an episode.  The
    model path is used only to fingerprint and validate the already imported,
    persistent Unreal asset.
    """
    episode.validate()
    if episode.joint_position is None:
        raise ValueError("URLab export requires an IK-solved episode")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    unknown_passes = set(required_passes) - set(CAMERA_NAMES)
    if unknown_passes:
        raise ValueError(f"Unknown URLab output passes: {sorted(unknown_passes)}")
    if "shadow" in required_passes:
        raise ValueError("The optional shadow-only pass is not yet production-qualified")
    if not set(DEFAULT_PASSES) <= set(required_passes):
        raise ValueError("URLab production jobs require rgba, depth_m, and instance_segmentation")
    if lumen_quality < 1:
        raise ValueError("Lumen quality must be positive")

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

    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    _validate_intrinsics(intrinsics)
    world_from_camera = np.stack(_camera_frames(camera, len(episode.timestamp))[:frame_count])
    _validate_rigid_transforms(world_from_camera)
    camera_pose = np.stack(
        [_opencv_camera_to_mujoco_pose(frame) for frame in world_from_camera]
    )
    arm_qpos = np.asarray(episode.joint_position[:frame_count], dtype=np.float64).reshape(
        frame_count, 14
    )
    finger_qpos = _gripper_qpos(episode.gripper[:frame_count])

    trajectory = destination / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        timestamp=np.asarray(episode.timestamp[:frame_count], dtype=np.float64),
        arm_qpos=arm_qpos,
        gripper_normalized=np.asarray(episode.gripper[:frame_count], dtype=np.float64),
        **(
            {"gripper_width_m": np.asarray(episode.gripper_width_m[:frame_count])}
            if episode.gripper_width_m is not None
            else {}
        ),
        finger_qpos=finger_qpos,
        world_from_camera=world_from_camera,
        # Retained for the current URLab mocap API and v1 reader compatibility.
        camera_pose_xyzw=camera_pose,
    )

    # This tiny wrapper is a compatibility/debug aid only.  It references the
    # source model in place and is not the asset loaded by production workers.
    compatibility_dir = destination / "model"
    compatibility_dir.mkdir(exist_ok=True)
    wrapper = compatibility_dir / "openarm_urlab.xml"
    wrapper.write_text(
        '<mujoco model="openarm_urlab_rerender">\n'
        f'  <include file="{model.as_posix()}"/>\n'
        "  <worldbody>\n"
        '    <body name="rerender_camera_rig" mocap="true">\n'
        '      <camera name="openarm_rgb" mode="fixed" '
        f'focalpixel="{intrinsics[0, 0]:.12g} {intrinsics[1, 1]:.12g}" '
        f'principalpixel="{intrinsics[0, 2]:.12g} {intrinsics[1, 2]:.12g}" '
        f'sensorsize="{width} {height}" resolution="{width} {height}"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )

    calibration_level = str(episode.metadata.get("calibration_level", "unvalidated"))
    calibration_validated = bool(episode.metadata.get("calibrated", False))
    payload: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "model_asset": {
            "identifier": URLAB_MODEL_ID,
            "revision": OPENARM_MUJOCO_COMMIT,
            "source_sha256": _sha256(model),
        },
        "trajectory_spec": {
            "path": trajectory.name,
            "sha256": _sha256(trajectory),
            "frames": frame_count,
            "arm_joint_order": [name for side in SIDES for name in ARM_JOINT_NAMES[side]],
            "finger_joint_order": [
                name for side in SIDES for name in FINGER_JOINT_NAMES[side]
            ],
            "gripper_semantics": "normalized 0=open, 1=closed",
            "pinch_center_compensated": bool(
                episode.metadata.get("pinch_center_compensated", False)
            ),
        },
        "camera": {
            "convention": "opencv-world-from-camera",
            "intrinsics": intrinsics.tolist(),
            "resolution": [width, height],
            "lens_distortion": "disabled-render-apply-in-post",
            "source": str(camera_path),
            "rig_mocap_body": "rerender_camera_rig",
            "streams": {name: CAMERA_NAMES[name] for name in required_passes},
        },
        "timing": {
            "fps": float(1.0 / episode.sample_period),
            "frame_range": [0, frame_count],
            "synchronous_capture": True,
            "motion_blur": False,
        },
        "render": {
            "lighting_preset": lighting_preset,
            "material_preset": material_preset,
            "lumen": {"enabled": True, "quality": lumen_quality},
            "hardware_ray_tracing": bool(hardware_ray_tracing),
            "exposure": {"mode": "manual", "compensation_ev": 0.0},
            "depth_unit": "metre",
            "alpha_source": "instance_segmentation",
            "required_passes": list(required_passes),
        },
        "random_seed": int(seed),
        "calibration": {
            "status": calibration_level,
            "validated": calibration_validated,
        },
        "provenance": {
            "source_dataset": episode.source_dataset,
            "source_episode": episode.source_episode,
            "task": episode.task,
            "urlab_commit": URLAB_COMMIT,
            "urlab_bridge_commit": URLAB_BRIDGE_COMMIT,
        },
        # Transitional keys keep existing automation readable while production
        # workers use only the structured v2 fields above.
        "frames": frame_count,
        "trajectory": str(trajectory),
        "fps": float(1.0 / episode.sample_period),
        "resolution": [width, height],
        "camera_rig_body": "rerender_camera_rig",
        "joint_order": [name for side in SIDES for name in ARM_JOINT_NAMES[side]],
        "registration_validated": calibration_validated,
        "render_contract": {
            "mode": "puppet",
            "physics_steps_per_frame": 0,
            "transport": "shm",
            "pixel_format": "RGBA8",
            "alpha_required": True,
        },
    }
    job = destination / "urlab_job.json"
    _write_json_atomic(job, payload)
    validate_urlab_job(job)
    return job


def _job_file(job_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else job_path.parent / path


def validate_urlab_job(job_path: str | Path) -> dict[str, Any]:
    """Fail closed on malformed, stale, or geometrically invalid v2 jobs."""
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    if job.get("schema") != JOB_SCHEMA:
        raise ValueError(f"Expected {JOB_SCHEMA}, got {job.get('schema')!r}")
    if job.get("model_asset", {}).get("identifier") != URLAB_MODEL_ID:
        raise ValueError("URLab job model identifier does not match the imported OpenArm asset")
    if job["model_asset"].get("revision") != OPENARM_MUJOCO_COMMIT:
        raise ValueError("URLab job model revision does not match the pinned OpenArm revision")
    if job.get("provenance", {}).get("urlab_commit") != URLAB_COMMIT:
        raise ValueError("URLab job plugin revision does not match this renderer")

    trajectory_spec = job["trajectory_spec"]
    trajectory_path = _job_file(job_path, trajectory_spec["path"])
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    if _sha256(trajectory_path) != trajectory_spec["sha256"]:
        raise ValueError("URLab trajectory checksum does not match the job manifest")
    expected_arms = [name for side in SIDES for name in ARM_JOINT_NAMES[side]]
    expected_fingers = [name for side in SIDES for name in FINGER_JOINT_NAMES[side]]
    if trajectory_spec.get("arm_joint_order") != expected_arms:
        raise ValueError("URLab arm joint ordering differs from the OpenArm contract")
    if trajectory_spec.get("finger_joint_order") != expected_fingers:
        raise ValueError("URLab finger joint ordering differs from the OpenArm contract")
    if trajectory_spec.get("gripper_semantics") != "normalized 0=open, 1=closed":
        raise ValueError("URLab gripper semantics differ from the OpenArm contract")

    frames = int(trajectory_spec["frames"])
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        required = {
            "timestamp": (frames,),
            "arm_qpos": (frames, 14),
            "gripper_normalized": (frames, 2),
            "finger_qpos": (frames, 4),
            "world_from_camera": (frames, 4, 4),
            "camera_pose_xyzw": (frames, 7),
        }
        for key, shape in required.items():
            if key not in trajectory or trajectory[key].shape != shape:
                actual = None if key not in trajectory else trajectory[key].shape
                raise ValueError(f"Trajectory {key} must have shape {shape}, got {actual}")
            if not np.all(np.isfinite(trajectory[key])):
                raise ValueError(f"Trajectory {key} contains non-finite values")
        if frames < 1 or (frames > 1 and np.any(np.diff(trajectory["timestamp"]) <= 0)):
            raise ValueError("Trajectory timestamps must be strictly increasing")
        if np.any((trajectory["gripper_normalized"] < 0) | (trajectory["gripper_normalized"] > 1)):
            raise ValueError("Normalized gripper values must be in [0,1]")
        if not np.allclose(
            trajectory["finger_qpos"],
            closure_to_finger_qpos(trajectory["gripper_normalized"]),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Finger qpos does not match normalized gripper closure")
        if "gripper_width_m" in trajectory:
            width_m = trajectory["gripper_width_m"]
            if width_m.shape != (frames, 2) or not np.all(np.isfinite(width_m)):
                raise ValueError("Physical gripper width must be finite with shape [frames,2]")
            if np.any(width_m < 0) or not np.allclose(
                trajectory["gripper_normalized"],
                aperture_to_closure(width_m),
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("Physical gripper width disagrees with normalized closure")
        _validate_rigid_transforms(trajectory["world_from_camera"])

    intrinsics = np.asarray(job["camera"]["intrinsics"], dtype=np.float64)
    _validate_intrinsics(intrinsics)
    width, height = map(int, job["camera"]["resolution"])
    if width <= 0 or height <= 0:
        raise ValueError("URLab resolution must be positive")
    frame_range = job["timing"]["frame_range"]
    if frame_range != [0, frames]:
        raise ValueError("URLab frame range must cover the exported trajectory")
    passes = set(job["render"]["required_passes"])
    if not set(DEFAULT_PASSES) <= passes or passes - set(CAMERA_NAMES):
        raise ValueError("URLab output pass contract is invalid")
    if job["render"].get("alpha_source") != "instance_segmentation":
        raise ValueError("Instance segmentation must be the authoritative alpha source")
    return {
        "ok": True,
        "schema": JOB_SCHEMA,
        "frames": frames,
        "trajectory": str(trajectory_path),
        "resolution": [width, height],
        "calibration_validated": bool(job["calibration"]["validated"]),
    }


def plan_urlab_shards(
    frame_count: int, shard_frames: int, warmup_frames: int
) -> list[dict[str, int]]:
    """Plan stable retained ranges with deterministic Lumen warm-up overlap."""
    if frame_count < 1 or shard_frames < 1 or warmup_frames < 0:
        raise ValueError("frame_count/shard_frames must be positive and warmup non-negative")
    result = []
    for start in range(0, frame_count, shard_frames):
        end = min(start + shard_frames, frame_count)
        result.append(
            {
                "warmup_start": max(0, start - warmup_frames),
                "retain_start": start,
                "retain_end": end,
            }
        )
    return result


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def prepare_urlab_asset(
    destination: str | Path = "data/assets/openarm_urlab",
    model_path: str | Path | None = None,
) -> Path:
    """Build the one persistent, self-contained MJCF import bundle."""
    model_path = resolve_model(model_path).resolve()
    destination = Path(destination).resolve()
    manifest_path = destination / "asset_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        wrapper = destination / existing.get("wrapper", "openarm_urlab.xml")
        if (
            existing.get("schema") == "openarm-urlab-asset-v2"
            and existing.get("color_conversion") == "sRGB IEC 61966-2-1 to Unreal linear"
            and existing.get("identifier") == URLAB_MODEL_ID
            and existing.get("source_sha256") == _sha256(model_path)
            and wrapper.is_file()
        ):
            return wrapper

    destination.mkdir(parents=True, exist_ok=True)
    # MuJoCo resolves compiler meshdir from the top-level wrapper. Keep the
    # wrapper beside the model's assets exactly once (never once per job).
    shutil.copytree(model_path.parent, destination, dirs_exist_ok=True)
    bundled_model = destination / model_path.name
    # MJCF colors in this pipeline are authored/interpreted as sRGB while
    # Unreal material vector parameters are linear. URLab currently assigns
    # the XML floats directly to FLinearColor, so convert the persistent import
    # copy once without changing the pinned source model.
    source_model = mujoco.MjModel.from_xml_path(str(model_path))
    xml_tree = ET.parse(bundled_model)
    for material in xml_tree.findall(".//material[@rgba]"):
        values = [float(value) for value in material.attrib["rgba"].split()]
        if len(values) >= 3:
            converted = [*[_srgb_to_linear(value) for value in values[:3]], *values[3:]]
            material.set("rgba", " ".join(f"{value:.9g}" for value in converted))
    xml_tree.write(bundled_model, encoding="unicode")
    wrapper = destination / "openarm_urlab.xml"
    cameras = "\n".join(
        f'      <camera name="{name}" mode="fixed" focalpixel="576 576" '
        'principalpixel="320 240" sensorsize="640 480" resolution="640 480"/>'
        for name in CAMERA_NAMES.values()
    )
    wrapper.write_text(
        '<mujoco model="openarm_urlab_rerender">\n'
        f'  <include file="{bundled_model.name}"/>\n'
        "  <worldbody>\n"
        '    <body name="rerender_camera_rig" mocap="true">\n'
        f"{cameras}\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )

    model = mujoco.MjModel.from_xml_path(str(wrapper))
    all_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    expected_arms = [name for side in SIDES for name in ARM_JOINT_NAMES[side]]
    expected_fingers = [name for side in SIDES for name in FINGER_JOINT_NAMES[side]]
    missing = [name for name in [*expected_arms, *expected_fingers] if name not in all_joints]
    if missing:
        raise ValueError(f"Persistent OpenArm import bundle is missing joints: {missing}")
    materials = []
    for index in range(source_model.nmat):
        name = (
            mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_MATERIAL, index)
            or f"material_{index}"
        )
        rgba = [float(value) for value in source_model.mat_rgba[index]]
        materials.append(
            {
                "name": name,
                "rgba_srgb": rgba,
                "rgba_unreal_linear": [*[_srgb_to_linear(value) for value in rgba[:3]], rgba[3]],
            }
        )
    _write_json_atomic(
        manifest_path,
        {
            "schema": "openarm-urlab-asset-v2",
            "identifier": URLAB_MODEL_ID,
            "model_revision": OPENARM_MUJOCO_COMMIT,
            "source": str(model_path),
            "source_sha256": _sha256(model_path),
            "wrapper": wrapper.name,
            "arm_joint_order": expected_arms,
            "finger_joint_order": expected_fingers,
            "arm_joint_count": len(expected_arms),
            "finger_joint_count": len(expected_fingers),
            "mesh_count": int(model.nmesh),
            "mesh_units": "compiled MuJoCo metres; preserve compiler scale on URLab import",
            "materials": materials,
            "color_conversion": "sRGB IEC 61966-2-1 to Unreal linear",
            "nanite_default": False,
            "nanite_audit_required": True,
            "camera_streams": CAMERA_NAMES,
        },
    )
    return wrapper


def import_urlab_asset(
    destination: str | Path = "data/assets/openarm_urlab",
    model_path: str | Path | None = None,
    *,
    address: str = "tcp://localhost",
) -> dict[str, Any]:
    """Import OpenArm once into the editor and save the sole render level."""
    try:
        from urlab_client import URLabAsset, URLabClient
    except ImportError as error:
        raise RuntimeError("Install the pinned URLab bridge before importing the asset") from error
    wrapper = prepare_urlab_asset(destination, model_path)
    with URLabClient(address) as client:
        client.discover(observations="minimal")
        handles = client.scene.apply_scene(
            "OpenArmRender",
            [URLabAsset(actor_id="openarm", xml=str(wrapper), location=(0, 0, 0))],
            save=True,
        )
        client.scene.spawn_light(
            "directional",
            actor_id="openarm_key",
            rotation_euler=(-35.0, -25.0, -35.0),
            intensity=4.0,
            color=(1.0, 0.96, 0.90),
        )
        client.scene.spawn_light(
            "directional",
            actor_id="openarm_fill",
            rotation_euler=(-20.0, 145.0, 20.0),
            intensity=1.5,
            color=(0.82, 0.90, 1.0),
        )
        client.scene.save_level()
        snapshot = client.scene.snapshot()
    return {
        "ok": True,
        "wrapper": str(wrapper),
        "level": "/Game/OpenArmRender",
        "actor_id": "openarm",
        "handle": str(handles["openarm"]),
        "snapshot": str(snapshot),
    }


def urlab_doctor(
    plugin_path: str | Path | None = None,
    project_path: str | Path = "unreal/OpenArmRenderer/OpenArmRenderer.uproject",
) -> dict[str, Any]:
    """Report independently verifiable prerequisites for editor and cooked workers."""
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
        Path(plugin_path).resolve()
        if plugin_path
        else Path("unreal/OpenArmRenderer/Plugins/UnrealRoboticsLab").resolve()
    )
    plugin_present = (plugin / "UnrealRoboticsLab.uplugin").exists()
    plugin_revision = None
    if (plugin / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(plugin), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        plugin_revision = result.stdout.strip() if result.returncode == 0 else None
    dependencies = {
        name: (plugin / "third_party" / "install" / name / "INSTALLED_SHA.txt").exists()
        for name in ("MuJoCo", "CoACD", "libzmq")
    }
    bridge_available = importlib.util.find_spec("urlab_client") is not None
    project_present = Path(project_path).is_file()
    ok = bool(
        editor
        and plugin_present
        and plugin_revision == URLAB_COMMIT
        and all(dependencies.values())
        and bridge_available
        and project_present
        and shutil.which("ffmpeg")
    )
    return {
        "ok": ok,
        "unreal_editor": editor,
        "required_unreal_version": "5.7",
        "plugin": str(plugin),
        "plugin_present": plugin_present,
        "plugin_revision": plugin_revision,
        "required_plugin_revision": URLAB_COMMIT,
        "dependencies": dependencies,
        "urlab_client_available": bridge_available,
        "required_bridge_revision": URLAB_BRIDGE_COMMIT,
        "project": str(Path(project_path).resolve()),
        "project_present": project_present,
        "ffmpeg": shutil.which("ffmpeg"),
        "gpu": shutil.which("nvidia-smi") is not None,
    }


def _atomic_imwrite(path: Path, image: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"Could not write {path}")
    temporary.replace(path)


class _FFmpegShardWriter:
    """Bounded asynchronous writer for lossless production pass videos."""

    def __init__(self, root: Path, width: int, height: int, fps: float, queue_size: int = 8):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise FileNotFoundError("ffmpeg is required for production URLab output")
        self.root = root
        self.paths = {
            "rgba": root / ".rgba.partial.mkv",
            "mask": root / ".instance.partial.mkv",
            "depth": root / ".depth_mm.partial.mkv",
        }

        def launch(pixel_format: str, path: Path) -> subprocess.Popen:
            return subprocess.Popen(
                [
                    executable,
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    pixel_format,
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    f"{fps:.12g}",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    str(path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        self.processes = {
            "rgba": launch("rgba", self.paths["rgba"]),
            "mask": launch("gray", self.paths["mask"]),
            "depth": launch("gray16le", self.paths["depth"]),
        }
        self.queue: queue.Queue[tuple[bytes, bytes, bytes] | None] = queue.Queue(
            maxsize=queue_size
        )
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="urlab-ffmpeg-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                for key, payload in zip(("rgba", "mask", "depth"), item, strict=True):
                    stream = self.processes[key].stdin
                    if stream is None:
                        raise RuntimeError(f"FFmpeg {key} stdin is unavailable")
                    stream.write(payload)
        except BaseException as error:
            self.error = error

    def put(self, rgba: np.ndarray, mask: np.ndarray, depth_m: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError("FFmpeg writer failed") from self.error
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        valid = np.isfinite(depth_m) & (depth_m > 0)
        depth_mm[valid] = np.rint(np.clip(depth_m[valid] * 1000.0, 1, 65535)).astype(np.uint16)
        self.queue.put((rgba.tobytes(), mask.tobytes(), depth_mm.astype("<u2").tobytes()))

    def close(self) -> list[Path]:
        self.queue.put(None)
        self.thread.join()
        failures = []
        for key, process in self.processes.items():
            if process.stdin is not None:
                process.stdin.close()
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            code = process.wait()
            if code != 0:
                failures.append(f"{key}: {stderr.strip()}")
        if self.error is not None:
            raise RuntimeError("FFmpeg writer thread failed") from self.error
        if failures:
            raise RuntimeError("FFmpeg encoding failed: " + "; ".join(failures))
        outputs = []
        names = {"rgba": "rgba.mkv", "mask": "instance.mkv", "depth": "depth_mm.mkv"}
        for key, partial in self.paths.items():
            final = self.root / names[key]
            partial.replace(final)
            outputs.append(final)
        return outputs


def _camera_lookup(client: Any, robot: Any, name: str) -> Any:
    camera = robot.cameras.get(name) if robot is not None else None
    camera = camera or client.global_cameras.get(name)
    if camera is None:
        available = sorted(
            set(client.global_cameras) | (set(robot.cameras) if robot is not None else set())
        )
        raise KeyError(f"URLab camera {name!r} is unavailable; found {available}")
    return camera


def render_urlab_job(
    job_path: str | Path,
    output_dir: str | Path,
    *,
    address: str = "tcp://localhost",
    transport: str = "shm",
    step_port: int = 5559,
    warmup_start: int = 0,
    retain_start: int = 0,
    retain_end: int | None = None,
    stop_editor_pie: bool = False,
    output_mode: str = "audit",
    writer_queue_size: int = 8,
) -> Path:
    """Replay a v2 job against a prepared editor or cooked URLab worker.

    RGB, metric depth, and instance segmentation are requested in the same
    synchronous step.  The RGB alpha byte is discarded and rebuilt exclusively
    from the OpenArm instance mask.
    """
    try:
        from urlab_client import URLabClient
    except ImportError as error:
        raise RuntimeError(
            "The pinned URLab Python bridge is not installed; run scripts/setup_urlab.sh"
        ) from error

    job_path = Path(job_path).resolve()
    validation = validate_urlab_job(job_path)
    job = json.loads(job_path.read_text())
    trajectory_path = Path(validation["trajectory"])
    trajectory = np.load(trajectory_path, allow_pickle=False)
    frame_count = int(validation["frames"])
    retain_end = frame_count if retain_end is None else retain_end
    if not (0 <= warmup_start <= retain_start < retain_end <= frame_count):
        raise ValueError("Invalid warm-up/retained URLab frame range")
    if output_mode not in {"audit", "production"}:
        raise ValueError("output_mode must be audit or production")

    output_dir = Path(output_dir).resolve()
    rgba_dir = output_dir / "rgba"
    depth_dir = output_dir / "depth"
    segmentation_dir = output_dir / "segmentation"
    if output_mode == "audit":
        for directory in (rgba_dir, depth_dir, segmentation_dir):
            directory.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    width, height = map(int, job["camera"]["resolution"])
    ffmpeg_writer = (
        _FFmpegShardWriter(
            output_dir,
            width,
            height,
            float(job["timing"]["fps"]),
            writer_queue_size,
        )
        if output_mode == "production"
        else None
    )

    started = time.perf_counter()
    captured_ids: list[int] = []
    plugin_version = None
    with URLabClient(
        address,
        step_mode="puppet",
        transport=transport,
        step_port=step_port,
    ) as client:
        client.discover(observations="minimal")
        started_pie = False
        if not client.manager_present:
            # Editor development path. Cooked workers already contain the one
            # imported OpenArm level and therefore skip all editor scene ops.
            client.sim.start(timeout_s=120.0)
            started_pie = True
        plugin_version = client.urlab_version
        robot = client.articulations_by_id.get("openarm") or client.articulations.get(
            "openarm_urlab_rerender"
        )
        if robot is None:
            available = sorted(client.articulations_by_id or client.articulations)
            raise RuntimeError(
                "Prepared URLab level is missing persistent actor_id='openarm'; "
                f"available={available}"
            )
        model = client.model
        if model is None or client.data is None:
            raise RuntimeError("URLab puppet mode did not provide a local MuJoCo model")
        client.reset(seed=int(job["random_seed"]))

        arm_addresses = []
        for name in job["trajectory_spec"]["arm_joint_order"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise KeyError(f"URLab model is missing arm joint {name}")
            arm_addresses.append(int(model.jnt_qposadr[joint_id]))
        finger_addresses = []
        for name in job["trajectory_spec"]["finger_joint_order"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise KeyError(f"URLab model is missing finger joint {name}")
            finger_addresses.append(int(model.jnt_qposadr[joint_id]))

        streams = job["camera"]["streams"]
        cameras = {key: _camera_lookup(client, robot, value) for key, value in streams.items()}
        include_cameras = {value: "sync" for value in streams.values()}
        expected_resolution = tuple(job["camera"]["resolution"])
        for frame in range(warmup_start, retain_end):
            client.data.qpos[arm_addresses] = trajectory["arm_qpos"][frame]
            client.data.qpos[finger_addresses] = trajectory["finger_qpos"][frame]
            camera_pose = trajectory["camera_pose_xyzw"][frame]
            client.runtime.set_mocap_pose(
                job["camera"]["rig_mocap_body"],
                pos=tuple(camera_pose[:3]),
                # scipy/export storage is xyzw; MuJoCo mocap RPC is wxyz.
                quat=(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]),
            )
            reply = client.step(
                n_steps=0,
                include_cameras=include_cameras,
                camera_query="fresh",
                camera_timeout_s=5.0,
                observations="minimal",
            )
            if getattr(reply, "cameras_stale", True):
                raise RuntimeError(f"URLab returned stale camera content at trajectory frame {frame}")
            step_frame_id = getattr(reply, "frame_id", None)
            camera_frame_ids = [camera.frame_id for camera in cameras.values()]
            if (
                step_frame_id is None
                or any(value is None for value in camera_frame_ids)
                or any(value != step_frame_id for value in camera_frame_ids)
            ):
                raise RuntimeError(
                    f"URLab pose/camera frame mismatch at {frame}: "
                    f"step={step_frame_id}, cameras={camera_frame_ids}"
                )
            if frame < retain_start:
                continue
            rgb = cameras["rgba"].latest_frame
            depth = cameras["depth_m"].latest_frame
            instance = cameras["instance_segmentation"].latest_frame
            if rgb is None or depth is None or instance is None:
                raise RuntimeError(f"URLab returned an incomplete capture at frame {frame}")
            rgb = np.asarray(rgb)
            depth = np.asarray(depth, dtype=np.float32)
            instance = np.asarray(instance)
            width, height = expected_resolution
            if rgb.shape != (height, width, 4):
                raise RuntimeError(f"RGB shape {rgb.shape} != {(height, width, 4)}")
            if depth.shape != (height, width) or instance.shape != (height, width, 4):
                raise RuntimeError(f"Depth/instance resolution mismatch at frame {frame}")

            # URLab segmentation is BGRA and only the OpenArm sibling pool is
            # visible.  Any non-black instance tint is therefore robot alpha.
            mask = np.any(instance[..., :3] != 0, axis=2).astype(np.uint8) * 255
            rgba = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA)
            rgba[..., 3] = mask
            # URLab's PF_R32 SceneDepth payload is in Unreal world units
            # (centimetres). Convert exactly once at the backend boundary.
            depth = depth.copy() * np.float32(0.01)
            depth[(mask == 0) | ~np.isfinite(depth) | (depth <= 0)] = np.inf

            if ffmpeg_writer is not None:
                # Convert OpenCV BGRA back to the declared raw RGBA input.
                ffmpeg_writer.put(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA), mask, depth)
            else:
                _atomic_imwrite(rgba_dir / f"{frame:06d}.png", rgba)
                _atomic_imwrite(segmentation_dir / f"{frame:06d}.png", mask)
                depth_tmp = depth_dir / f".{frame:06d}.tmp.npz"
                np.savez_compressed(depth_tmp, depth_m=depth)
                depth_tmp.replace(depth_dir / f"{frame:06d}.npz")
            # The sync request is the hard anti-lag boundary. Preserve the
            # server's step frame id when the bridge revision exposes it.
            captured_ids.append(int(step_frame_id))

        if started_pie and stop_editor_pie:
            client.sim.stop()
    trajectory.close()

    encoded_files = ffmpeg_writer.close() if ffmpeg_writer is not None else []

    elapsed = time.perf_counter() - started
    files = sorted(
        encoded_files
        or [*rgba_dir.glob("*.png"), *depth_dir.glob("*.npz"), *segmentation_dir.glob("*.png")]
    )
    manifest = output_dir / "urlab_render_manifest.json"
    _write_json_atomic(
        manifest,
        {
            "schema": "openarm-urlab-render-shard-v2",
            "status": "complete",
            "job": str(job_path),
            "job_sha256": _sha256(job_path),
            "urlab_plugin_version": plugin_version,
            "warmup_range": [warmup_start, retain_start],
            "retained_range": [retain_start, retain_end],
            "frames": retain_end - retain_start,
            "synchronous_frame_ids": captured_ids,
            "elapsed_seconds": elapsed,
            "frames_per_second": float((retain_end - warmup_start) / elapsed),
            "transport": transport,
            "address": address,
            "step_port": step_port,
            "depth_unit": "metre",
            "alpha_source": "instance_segmentation",
            "output_mode": output_mode,
            "depth_video_encoding": (
                "lossless FFV1 gray16le millimetres; zero means invalid"
                if output_mode == "production"
                else None
            ),
            "checksums": {str(path.relative_to(output_dir)): _sha256(path) for path in files},
            "failures": [],
        },
    )
    return manifest


def render_urlab_batch(
    job_path: str | Path,
    output_dir: str | Path,
    *,
    gpu_ids: tuple[int, ...] = (0,),
    shard_frames: int = 256,
    warmup_frames: int = 8,
    resume: bool = True,
    transport: str = "shm",
    address: str = "tcp://localhost",
    base_step_port: int = 5559,
    runtime: str | Path | None = None,
    startup_timeout_s: float = 120.0,
    output_mode: str = "production",
    writer_queue_size: int = 8,
) -> Path:
    """Render deterministic shards concurrently against one prepared worker per GPU.

    Workers use distinct URLab RPC ports (`base_step_port + worker index`). A
    cooked runtime can be launched separately with the matching
    `-URLabPortOffset` argument; keeping process lifecycle separate also permits
    workers on different hosts while retaining the same shard contract.
    """
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    validation = validate_urlab_job(job_path)
    job_path = Path(job_path).resolve()
    output_dir = Path(output_dir).resolve()
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shards = plan_urlab_shards(validation["frames"], shard_frames, warmup_frames)
    pending: list[tuple[int, dict[str, int], Path]] = []
    for index, shard in enumerate(shards):
        name = f"{shard['retain_start']:06d}_{shard['retain_end']:06d}"
        final = shard_root / name
        manifest = final / "urlab_render_manifest.json"
        if resume and manifest.is_file():
            try:
                existing = json.loads(manifest.read_text())
                checksums_valid = all(
                    (final / relative).is_file()
                    and _sha256(final / relative) == checksum
                    for relative, checksum in existing.get("checksums", {}).items()
                )
                if (
                    existing.get("status") == "complete"
                    and existing.get("job_sha256") == _sha256(job_path)
                    and bool(existing.get("checksums"))
                    and checksums_valid
                ):
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        pending.append((index, shard, final))

    started = time.perf_counter()
    processes: list[tuple[subprocess.Popen, Any]] = []
    if runtime is not None:
        runtime = Path(runtime).resolve()
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            raise FileNotFoundError(f"Cooked OpenArm runtime is not executable: {runtime}")
        host = address.removeprefix("tcp://").split(":", 1)[0]
        if host in {"localhost", "0.0.0.0"}:
            host = "127.0.0.1"
        for worker, gpu_id in enumerate(gpu_ids):
            worker_dir = output_dir / "workers" / f"gpu_{gpu_id}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            log = (worker_dir / "runtime.log").open("w")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment["OPENARM_URLAB_JOB"] = str(job_path)
            port_offset = base_step_port - 5559 + worker
            process = subprocess.Popen(
                [
                    str(runtime),
                    f"-OpenArmJob={job_path}",
                    f"-URLabPortOffset={port_offset}",
                    f"-UserDir={worker_dir}",
                    "-RenderOffscreen",
                    "-Unattended",
                    "-NoSound",
                    "-NoSplash",
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, log))
        try:
            deadline = time.monotonic() + startup_timeout_s
            for worker, (process, _) in enumerate(processes):
                port = base_step_port + worker
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"URLab worker {worker} exited during startup with code {process.returncode}"
                        )
                    try:
                        with socket.create_connection((host, port), timeout=0.25):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"URLab worker {worker} did not bind {host}:{port}")
                        time.sleep(0.1)
        except Exception:
            for process, log in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                log.close()
            raise

    def run(item: tuple[int, dict[str, int], Path]) -> Path:
        index, shard, final = item
        worker = index % len(gpu_ids)
        partial = final.with_name(f".{final.name}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            manifest = render_urlab_job(
                job_path,
                partial,
                address=address,
                transport=transport,
                step_port=base_step_port + worker,
                warmup_start=shard["warmup_start"],
                retain_start=shard["retain_start"],
                retain_end=shard["retain_end"],
                output_mode=output_mode,
                writer_queue_size=writer_queue_size,
            )
            if final.exists():
                shutil.rmtree(final)
            partial.replace(final)
            return final / manifest.name
        except Exception as error:
            _write_json_atomic(
                partial / "failure_manifest.json",
                {
                    "schema": "openarm-urlab-render-shard-v2",
                    "status": "failed",
                    "job": str(job_path),
                    "range": shard,
                    "gpu_id": gpu_ids[worker],
                    "step_port": base_step_port + worker,
                    "failure": f"{type(error).__name__}: {error}",
                },
            )
            raise

    # Each worker handles one shard at a time. Stable modulo assignment keeps
    # resume and temporal boundary behaviour independent of scheduling order.
    worker_groups = [
        [item for item in pending if item[0] % len(gpu_ids) == worker]
        for worker in range(len(gpu_ids))
    ]

    def run_group(group: list[tuple[int, dict[str, int], Path]]) -> list[Path]:
        return [run(item) for item in group]

    try:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(run_group, group) for group in worker_groups if group]
            for future in futures:
                future.result()
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            log.close()

    manifests = sorted(shard_root.glob("*/urlab_render_manifest.json"))
    elapsed = time.perf_counter() - started
    result = output_dir / "render_manifest.json"
    _write_json_atomic(
        result,
        {
            "schema": "openarm-urlab-render-batch-v2",
            "status": "complete",
            "job": str(job_path),
            "job_sha256": _sha256(job_path),
            "frames": validation["frames"],
            "gpu_ids": list(gpu_ids),
            "workers": len(gpu_ids),
            "runtime": str(runtime) if runtime is not None else None,
            "output_mode": output_mode,
            "writer_queue_size": writer_queue_size,
            "shard_frames": shard_frames,
            "warmup_frames": warmup_frames,
            "shards": [str(path.relative_to(output_dir)) for path in manifests],
            "new_shards": len(pending),
            "resumed_shards": len(shards) - len(pending),
            "elapsed_seconds": elapsed,
            "failures": [],
        },
    )
    return result


def _rgba_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected RGBA image: {path}")
    return image[..., 3] > 0


def _symmetric_boundary_distance_px(first: np.ndarray, second: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    first_u8 = first.astype(np.uint8)
    second_u8 = second.astype(np.uint8)
    first_edge = first_u8 ^ cv2.erode(first_u8, kernel)
    second_edge = second_u8 ^ cv2.erode(second_u8, kernel)
    if not np.any(first_edge) or not np.any(second_edge):
        return 0.0 if np.array_equal(first, second) else float("inf")
    distance_to_first = cv2.distanceTransform(1 - first_edge, cv2.DIST_L2, 5)
    distance_to_second = cv2.distanceTransform(1 - second_edge, cv2.DIST_L2, 5)
    return float(
        0.5
        * (
            np.mean(distance_to_second[first_edge.astype(bool)])
            + np.mean(distance_to_first[second_edge.astype(bool)])
        )
    )


def validate_urlab_against_references(
    urlab_rgba: str | Path,
    blender_rgba: str | Path,
    mujoco_rgba: str | Path | None = None,
    *,
    minimum_mean_iou: float = 0.95,
    minimum_p05_iou: float = 0.90,
    maximum_mean_boundary_error_px: float = 1.0,
) -> dict[str, Any]:
    """Compare URLab silhouettes with Blender and optional MuJoCo references."""
    candidate = sorted(Path(urlab_rgba).glob("*.png"))
    references = {"blender": sorted(Path(blender_rgba).glob("*.png"))}
    if mujoco_rgba is not None:
        references["mujoco"] = sorted(Path(mujoco_rgba).glob("*.png"))
    if not candidate:
        raise ValueError("No URLab RGBA frames found")
    errors: list[str] = []
    metrics: dict[str, Any] = {}
    candidate_masks = [_rgba_mask(path) for path in candidate]
    for label, paths in references.items():
        if len(paths) != len(candidate):
            errors.append(f"{label} frame count {len(paths)} != URLab {len(candidate)}")
            continue
        ious = []
        boundary_errors = []
        for generated, path in zip(candidate_masks, paths, strict=True):
            reference = _rgba_mask(path)
            if reference.shape != generated.shape:
                raise ValueError(f"Reference resolution mismatch: {path}")
            union = generated | reference
            ious.append(float(np.count_nonzero(generated & reference) / max(1, np.count_nonzero(union))))
            boundary_errors.append(_symmetric_boundary_distance_px(generated, reference))
        mean_iou = float(np.mean(ious))
        p05_iou = float(np.quantile(ious, 0.05))
        mean_boundary_error = float(np.mean(boundary_errors))
        metrics[label] = {
            "mean_iou": mean_iou,
            "p05_iou": p05_iou,
            "minimum_iou": float(np.min(ious)),
            "mean_boundary_error_px": mean_boundary_error,
            "p95_boundary_error_px": float(np.quantile(boundary_errors, 0.95)),
        }
        if mean_iou < minimum_mean_iou:
            errors.append(f"{label} mean IoU {mean_iou:.4f} < {minimum_mean_iou:.4f}")
        if p05_iou < minimum_p05_iou:
            errors.append(f"{label} p05 IoU {p05_iou:.4f} < {minimum_p05_iou:.4f}")
        if mean_boundary_error > maximum_mean_boundary_error_px:
            errors.append(
                f"{label} mean boundary error {mean_boundary_error:.3f}px > "
                f"{maximum_mean_boundary_error_px:.3f}px"
            )
    return {
        "ok": not errors,
        "schema": "openarm-urlab-reference-validation-v1",
        "frames": len(candidate),
        "metrics": metrics,
        "required_mean_iou": minimum_mean_iou,
        "required_p05_iou": minimum_p05_iou,
        "maximum_mean_boundary_error_px": maximum_mean_boundary_error_px,
        "errors": errors,
    }
