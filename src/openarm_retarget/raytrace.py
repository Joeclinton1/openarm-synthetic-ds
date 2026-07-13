from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from .constants import ARM_JOINT_NAMES, OPENARM_MUJOCO_COMMIT, SIDES
from .model import resolve_model
from .schema import Episode


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )


def _write_compiled_mesh(model: mujoco.MjModel, mesh_id: int, destination: Path) -> None:
    """Write MuJoCo's compiled mesh, preserving compiler-applied scale and centering."""
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
    faces = model.mesh_face[face_start : face_start + face_count]
    with destination.open("w") as stream:
        stream.write("# Compiled from the official OpenArm 2.0 MuJoCo model\n")
        for vertex in vertices:
            stream.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            stream.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _material(model: mujoco.MjModel, geom_id: int) -> dict:
    material_id = int(model.geom_matid[geom_id])
    if material_id >= 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
        rgba = model.mat_rgba[material_id]
        reflectance = float(model.mat_reflectance[material_id])
        shininess = float(model.mat_shininess[material_id])
    else:
        name = "geom_rgba"
        rgba = model.geom_rgba[geom_id]
        reflectance = 0.0
        shininess = 0.5
    return {
        "name": name,
        "rgba": [float(value) for value in rgba],
        "color_space": "sRGB",
        "metallic": min(1.0, max(0.0, reflectance)),
        "roughness": min(1.0, max(0.04, 1.0 - shininess)),
    }


def _default_camera(width: int, height: int) -> dict:
    # A documented preview camera. Source-aligned production renders should pass camera_json.
    position = np.array([1.05, -1.05, 0.62])
    target = np.array([0.0, 0.0, -0.18])
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = np.column_stack([right, down, forward])
    world_from_camera[:3, 3] = position
    focal = 0.9 * width
    return {
        "intrinsics": [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        "world_from_camera_frames": [world_from_camera.tolist()],
        "source": "unvalidated preview camera",
    }


def export_blender_scene(
    episode: Episode,
    destination: str | Path,
    model_path: str | Path | None = None,
    camera_json: str | Path | None = None,
    width: int = 960,
    height: int = 720,
    max_frames: int | None = None,
    samples: int = 32,
    eevee_samples: int = 16,
    png_compression: int = 15,
) -> Path:
    """Export official OpenArm visual meshes and per-frame transforms for Blender Cycles."""
    if episode.joint_position is None:
        raise ValueError("Blender export requires an IK-solved episode")
    if width <= 0 or height <= 0 or samples < 0 or eevee_samples < 1:
        raise ValueError(
            "Dimensions/EEVEE samples must be positive; Cycles samples cannot be negative"
        )
    if not 0 <= png_compression <= 100:
        raise ValueError("PNG compression must be in [0,100]")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    asset_dir = destination / "meshes"
    asset_dir.mkdir(exist_ok=True)

    resolved_model = resolve_model(model_path)
    model = mujoco.MjModel.from_xml_path(str(resolved_model))
    data = mujoco.MjData(model)
    qpos = {}
    for side in SIDES:
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES[side]
        ]
        qpos[side] = model.jnt_qposadr[joint_ids]

    visual_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) == 2
        and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    ]
    objects = []
    for geom_id in visual_geoms:
        raw_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        name = _safe_name(raw_name or f"geom_{geom_id}")
        mesh_path = asset_dir / f"{name}.obj"
        _write_compiled_mesh(model, int(model.geom_dataid[geom_id]), mesh_path)
        objects.append(
            {
                "name": name,
                "geom_id": geom_id,
                "mesh": str(mesh_path.relative_to(destination)),
                "material": _material(model, geom_id),
                "world_from_object_frames": [],
            }
        )

    frame_count = (
        len(episode.timestamp) if max_frames is None else min(max_frames, len(episode.timestamp))
    )
    if frame_count < 1:
        raise ValueError("Scene must contain at least one frame")
    for frame in range(frame_count):
        for side_index, side in enumerate(SIDES):
            data.qpos[qpos[side]] = episode.joint_position[frame, side_index]
        mujoco.mj_forward(model, data)
        for item in objects:
            geom_id = item["geom_id"]
            transform = np.eye(4)
            transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
            transform[:3, 3] = data.geom_xpos[geom_id]
            item["world_from_object_frames"].append(transform.tolist())
    for item in objects:
        del item["geom_id"]

    if camera_json:
        camera = json.loads(Path(camera_json).read_text())
        camera["source"] = str(Path(camera_json).resolve())
    else:
        camera = _default_camera(width, height)
    poses = camera.get("world_from_camera_frames") or [camera.get("world_from_camera")]
    if not poses or poses[0] is None:
        raise ValueError("Camera JSON requires world_from_camera or world_from_camera_frames")
    if len(poses) not in (1, len(episode.timestamp), frame_count):
        raise ValueError("Camera trajectory length does not match the episode")
    camera["world_from_camera_frames"] = poses[:frame_count] if len(poses) > 1 else poses

    payload = {
        "schema": "openarm-blender-cycles-scene-v1",
        "engine": "BLENDER_EEVEE_NEXT" if samples == 0 else "CYCLES",
        "model": str(resolved_model),
        "model_revision": OPENARM_MUJOCO_COMMIT,
        "episode_frames": frame_count,
        "fps": float(1.0 / episode.sample_period),
        "resolution": [width, height],
        "samples": samples,
        "eevee_samples": eevee_samples if samples == 0 else None,
        "png_compression": png_compression,
        "transparent_background": True,
        "lighting": {
            "world_color_linear": [0.12, 0.12, 0.12, 1.0],
            "world_strength": 0.35,
            "area_lights": [
                {"location": [1.4, -1.0, 1.8], "energy_w": 160.0, "size_m": 2.0},
                {"location": [-1.2, 0.8, 1.2], "energy_w": 100.0, "size_m": 1.5},
            ],
            "rationale": (
                "neutral low-key studio illumination preserving official OpenArm v2 "
                "matte-black and silver material separation"
            ),
        },
        "camera": camera,
        "objects": objects,
        "registration_validated": bool(episode.metadata.get("calibrated", False)),
    }
    output = destination / "scene.json"
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return output


def render_blender_scene(
    scene: str | Path,
    output: str | Path,
    blender: str | Path = "blender",
    max_frames: int | None = None,
    device: str = "CPU",
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    resume: bool = False,
    gpu_id: int | None = None,
    write_depth: bool = False,
) -> Path:
    """Run the bundled deterministic Blender driver in background mode."""
    executable = shutil.which(str(blender)) or str(Path(blender).resolve())
    if not Path(executable).exists():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    scene = Path(scene).resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name("blender_driver.py").resolve()
    command = [
        executable,
        "--background",
        "--factory-startup",
        "--python",
        str(driver),
        "--",
        str(scene),
        str(output),
        "--device",
        device,
        "--start-frame",
        str(start_frame),
    ]
    if end_frame is not None:
        command.extend(["--end-frame", str(end_frame)])
    if resume:
        command.append("--resume")
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])
    if write_depth:
        command.append("--write-depth")
    environment = os.environ.copy()
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = output / f"blender_{start_frame:06d}_{(end_frame or 0):06d}.log"
    with log_path.open("w") as log:
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if "Traceback (most recent call last)" in log_path.read_text(errors="replace"):
        raise RuntimeError(f"Blender driver failed; see {log_path}")
    return output


def _frame_ranges(frame_count: int, workers: int) -> list[tuple[int, int]]:
    if frame_count < 1 or workers < 1:
        raise ValueError("frame_count and workers must be positive")
    workers = min(frame_count, workers)
    base, extra = divmod(frame_count, workers)
    ranges = []
    start = 0
    for index in range(workers):
        end = start + base + (1 if index < extra else 0)
        ranges.append((start, end))
        start = end
    return ranges


def _blender_identity(blender: str | Path) -> dict[str, str]:
    executable = shutil.which(str(blender)) or str(Path(blender).resolve())
    result = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True
    )
    return {"executable": executable, "version": result.stdout.splitlines()[0].strip()}


def render_blender_batch(
    scene: str | Path,
    output: str | Path,
    blender: str | Path = "blender",
    *,
    device: str = "OPTIX",
    gpu_ids: tuple[int, ...] = (0,),
    workers: int | None = None,
    resume: bool = True,
    max_frames: int | None = None,
    write_depth: bool = False,
) -> Path:
    """Render disjoint frame ranges concurrently and write a resumable benchmark manifest."""
    scene = Path(scene).resolve()
    spec = json.loads(scene.read_text())
    frame_count = int(spec["episode_frames"])
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if workers is None:
        # EEVEE uses one graphics context and does not scale through Cycles' CUDA/OptiX
        # device selection. On the measured host, two EEVEE processes halved throughput.
        workers = 1 if spec.get("engine") == "BLENDER_EEVEE_NEXT" else len(gpu_ids)
        if device == "CPU":
            workers = 1
    if workers < 1:
        raise ValueError("workers must be positive")
    if device != "CPU" and not gpu_ids:
        raise ValueError("At least one GPU id is required for GPU rendering")
    ranges = _frame_ranges(frame_count, workers)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = {
        path.name for path in output.glob("*.png") if path.is_file() and path.stat().st_size > 100
    }
    expected = {f"{frame:06d}.png" for frame in range(frame_count)}
    depth_root = output / "depth"
    expected_depth = {f"{frame:06d}.npz" for frame in range(frame_count)}
    before_depth = (
        {path.name for path in depth_root.glob("*.npz") if path.stat().st_size > 100}
        if write_depth
        else set()
    )
    started = time.perf_counter()

    # Do not pay Blender's scene-import and shader-compilation cost when a resumed
    # range is already complete. This also makes completed jobs cheap to audit/restart.
    if resume and expected <= before and (not write_depth or expected_depth <= before_depth):
        manifest = {
            "schema": "openarm-blender-render-batch-v1",
            "scene": str(scene),
            "engine": spec.get("engine", "CYCLES"),
            "device": device,
            "gpu_ids": list(gpu_ids),
            "workers": len(ranges),
            "frame_ranges": [list(value) for value in ranges],
            "frames": frame_count,
            "new_frames": 0,
            "resumed_frames": frame_count,
            "seconds": time.perf_counter() - started,
            "effective_fps": 0.0,
            "resolution": spec["resolution"],
            "samples": spec["samples"],
            "missing_frames": [],
            "depth_pass": write_depth,
            "missing_depth_frames": [],
            "resume_noop": True,
        }
        path = output / "render_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path

    def run(worker_index: int, frame_range: tuple[int, int]) -> Path:
        gpu_id = gpu_ids[worker_index % len(gpu_ids)] if device != "CPU" else None
        return render_blender_scene(
            scene,
            output,
            blender,
            device=device,
            start_frame=frame_range[0],
            end_frame=frame_range[1],
            resume=resume,
            gpu_id=gpu_id,
            write_depth=write_depth,
        )

    blender_identity = _blender_identity(blender)
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = [pool.submit(run, index, frame_range) for index, frame_range in enumerate(ranges)]
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - started
    after = {
        path.name for path in output.glob("*.png") if path.is_file() and path.stat().st_size > 100
    }
    missing = sorted(expected - after)
    if missing:
        raise RuntimeError(f"Blender batch is missing {len(missing)} frames; first: {missing[0]}")
    rendered = len((after - before) & expected)
    after_depth = (
        {path.name for path in depth_root.glob("*.npz") if path.stat().st_size > 100}
        if write_depth
        else set()
    )
    missing_depth = sorted(expected_depth - after_depth) if write_depth else []
    if missing_depth:
        raise RuntimeError(
            f"Blender batch is missing {len(missing_depth)} depth frames; first: {missing_depth[0]}"
        )
    manifest = {
        "schema": "openarm-blender-render-batch-v1",
        "scene": str(scene),
        "blender": blender_identity,
        "engine": spec.get("engine", "CYCLES"),
        "device": device,
        "gpu_ids": list(gpu_ids),
        "workers": len(ranges),
        "frame_ranges": [list(value) for value in ranges],
        "frames": frame_count,
        "new_frames": rendered,
        "resumed_frames": frame_count - rendered,
        "seconds": elapsed,
        "effective_fps": rendered / elapsed if elapsed else 0.0,
        "resolution": spec["resolution"],
        "samples": spec["samples"],
        "missing_frames": missing,
        "depth_pass": write_depth,
        "missing_depth_frames": missing_depth,
        "resume_noop": False,
    }
    path = output / "render_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
