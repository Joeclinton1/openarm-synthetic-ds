from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .constants import ARM_JOINT_NAMES, SIDES
from .export import validate_lerobot_v3
from .filters import FilterConfig
from .model import resolve_model
from .schema import Episode


def audit_conversion(destination: str | Path, model_path: str | Path | None = None) -> dict:
    """Independently check canonical poses, IK constraints, shared frames, and LeRobot export."""
    destination = Path(destination)
    quality = json.loads((destination / "quality_report.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(resolve_model(model_path)))
    limits = {}
    for side in SIDES:
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES[side]
        ]
        limits[side] = model.jnt_range[joint_ids]

    config = FilterConfig()
    errors: list[str] = []
    episode_files = sorted((destination / "episodes").glob("*.npz"))
    input_frames = feasible_frames = 0
    maximums = {
        "quaternion_norm_error": 0.0,
        "position_error_m": 0.0,
        "orientation_error_rad": 0.0,
        "jacobian_condition": 0.0,
        "joint_velocity_rad_s": 0.0,
        "joint_acceleration_rad_s2": 0.0,
        "joint_limit_violation_rad": 0.0,
    }
    feasible_collisions = 0
    for path in episode_files:
        episode = Episode.load(path)
        input_frames += len(episode.timestamp)
        if episode.feasible is None or episode.joint_position is None:
            errors.append(f"{path.name}: missing IK or feasibility output")
            continue
        feasible = episode.feasible
        feasible_frames += int(feasible.sum())
        norms = np.linalg.norm(episode.ee_pose[..., 3:], axis=-1)
        maximums["quaternion_norm_error"] = max(
            maximums["quaternion_norm_error"], float(np.max(np.abs(norms - 1.0)))
        )
        if not np.isfinite(episode.ee_pose).all() or not np.isfinite(episode.joint_position).all():
            errors.append(f"{path.name}: non-finite canonical or joint value")
        registration = episode.metadata.get("registration", {})
        bases = registration.get("openarm_from_source_base", {})
        if registration:
            right_base = np.asarray(bases.get("right", []))
            left_base = np.asarray(bases.get("left", []))
            if (
                not registration.get("shared_base_frame")
                or right_base.shape != (7,)
                or left_base.shape != (7,)
                or not np.allclose(right_base, left_base, atol=1e-10)
            ):
                errors.append(f"{path.name}: right/left base frames differ or are malformed")
        for side_index, side in enumerate(SIDES):
            lower, upper = limits[side][:, 0], limits[side][:, 1]
            positions = episode.joint_position[:, side_index]
            violation = np.maximum(lower - positions, positions - upper)
            maximums["joint_limit_violation_rad"] = max(
                maximums["joint_limit_violation_rad"], float(max(0.0, np.max(violation)))
            )
        active = [SIDES.index(side) for side in episode.metadata.get("active_sides", SIDES)]
        diagnostics = episode.diagnostics
        for key in (
            "position_error_m",
            "orientation_error_rad",
            "jacobian_condition",
            "joint_velocity_rad_s",
            "joint_acceleration_rad_s2",
        ):
            values = np.asarray(diagnostics[key])
            selected = values[feasible][:, active]
            if selected.size:
                maximums[key] = max(maximums[key], float(np.max(np.abs(selected))))
        feasible_collisions += int(np.count_nonzero(diagnostics["invalid_collision"][feasible]))

    thresholds = {
        "quaternion_norm_error": 1e-5,
        "position_error_m": config.max_position_error_m + 1e-8,
        "orientation_error_rad": config.max_orientation_error_rad + 1e-8,
        "jacobian_condition": config.max_jacobian_condition + 1e-6,
        "joint_velocity_rad_s": config.max_joint_velocity_rad_s + 1e-6,
        "joint_acceleration_rad_s2": config.max_joint_acceleration_rad_s2 + 1e-6,
        "joint_limit_violation_rad": 1e-7,
    }
    for key, threshold in thresholds.items():
        if maximums[key] > threshold:
            errors.append(f"{key} {maximums[key]:.8g} exceeds {threshold:.8g}")
    if feasible_collisions:
        errors.append(f"{feasible_collisions} feasible frames contain a collision")
    if input_frames != int(quality["input_frames"]):
        errors.append("quality input frame count differs from episode files")
    if feasible_frames != int(quality["feasible_frames"]):
        errors.append("quality feasible frame count differs from episode files")
    export = validate_lerobot_v3(destination / "lerobot")
    if not export["ok"]:
        errors.extend(f"export: {item}" for item in export["errors"])
    if export.get("total_frames") != feasible_frames:
        errors.append("LeRobot rows differ from feasible frame count")
    return {
        "ok": not errors,
        "destination": str(destination.resolve()),
        "source_repo_id": quality["source_repo_id"],
        "source_revision": quality["source_revision"],
        "input_frames": input_frames,
        "input_seconds": quality["input_seconds"],
        "feasible_frames": feasible_frames,
        "feasible_fraction": feasible_frames / input_frames,
        "shared_base_frame": quality.get("shared_base_frame", False),
        "calibration_validated": quality.get("calibration_validated", False),
        "maximum_feasible_diagnostics": maximums,
        "feasible_collisions": feasible_collisions,
        "lerobot": export,
        "errors": errors,
    }


def audit_all(
    destinations: list[str | Path],
    output: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict:
    datasets = [audit_conversion(destination, model_path) for destination in destinations]
    report = {
        "ok": all(dataset["ok"] for dataset in datasets),
        "kinematic_acceptance": all(dataset["ok"] for dataset in datasets),
        "physical_release_ready": all(dataset["calibration_validated"] for dataset in datasets),
        "datasets": datasets,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report
