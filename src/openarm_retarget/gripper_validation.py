from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import SIDES
from .gripper import (
    OPENARM_MAX_APERTURE_M,
    OpenArmPinchKinematics,
    closure_to_aperture,
    pinch_midpoint_local,
)
from .poses import pose_to_matrix
from .schema import Episode


def _summary(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def measure_gripper_contact(
    episode: Episode, model_path: str | Path | None = None
) -> dict[str, np.ndarray | str]:
    """Measure the exact fingertip state without applying acceptance policy."""
    episode.validate()
    if episode.joint_position is None:
        raise ValueError("Gripper contact measurement requires an IK-solved episode")
    kinematics = OpenArmPinchKinematics(model_path)
    frames = len(episode.timestamp)
    actual_midpoint = np.zeros((frames, 2, 3), dtype=np.float64)
    actual_aperture = np.zeros((frames, 2), dtype=np.float64)
    for frame in range(frames):
        kinematics.set_frame(episode.joint_position[frame], episode.gripper[frame])
        for side_index, side in enumerate(SIDES):
            actual_midpoint[frame, side_index], actual_aperture[frame, side_index] = (
                kinematics.measure(side)
            )

    if "pinch_center_target_m" in episode.diagnostics:
        target_midpoint = np.asarray(episode.diagnostics["pinch_center_target_m"], dtype=np.float64)
        target_kind = "source pinch center preserved across gripper motion"
    else:
        transforms = pose_to_matrix(episode.ee_pose)
        local_midpoint = pinch_midpoint_local(episode.gripper)
        target_midpoint = transforms[..., :3, 3] + np.einsum(
            "...ij,...j->...i", transforms[..., :3, :3], local_midpoint
        )
        target_kind = "EE target plus official moving-finger geometry"
    if target_midpoint.shape != (frames, 2, 3):
        raise ValueError("pinch_center_target_m must have shape [T,2,3]")
    expected_aperture = closure_to_aperture(episode.gripper)
    return {
        "actual_midpoint_m": actual_midpoint,
        "target_midpoint_m": target_midpoint,
        "pinch_error_m": np.linalg.norm(actual_midpoint - target_midpoint, axis=-1),
        "actual_aperture_m": actual_aperture,
        "expected_aperture_m": expected_aperture,
        "aperture_error_m": np.abs(actual_aperture - expected_aperture),
        "target_kind": target_kind,
    }


def validate_gripper_contact(
    episode: Episode,
    model_path: str | Path | None = None,
    *,
    maximum_pinch_error_m: float = 0.01,
    maximum_aperture_error_m: float = 1e-6,
) -> dict:
    """Validate fingertip midpoint and opening for the exact rendered frame state."""
    episode.validate()
    if episode.joint_position is None:
        raise ValueError("Gripper contact validation requires an IK-solved episode")
    if maximum_pinch_error_m <= 0 or maximum_aperture_error_m <= 0:
        raise ValueError("Validation tolerances must be positive")

    frames = len(episode.timestamp)
    measurement = measure_gripper_contact(episode, model_path)
    actual_aperture = np.asarray(measurement["actual_aperture_m"])
    pinch_error = np.asarray(measurement["pinch_error_m"])
    aperture_error = np.asarray(measurement["aperture_error_m"])
    active = set(episode.metadata.get("active_sides", SIDES))
    if episode.feasible is not None:
        accepted = np.broadcast_to(np.asarray(episode.feasible, dtype=bool)[:, None], (frames, 2))
    else:
        accepted = np.asarray(
            episode.diagnostics.get("ik_success", np.ones((frames, 2), dtype=bool)), dtype=bool
        )

    side_reports = {}
    all_valid = []
    for side_index, side in enumerate(SIDES):
        valid = accepted[:, side_index] if side in active else np.zeros(frames, dtype=bool)
        all_valid.append(valid)
        side_reports[side] = {
            "active": side in active,
            "validated_frames": int(np.sum(valid)),
            "pinch_error_m": _summary(pinch_error[valid, side_index]),
            "aperture_error_m": _summary(aperture_error[valid, side_index]),
        }
    valid_values = np.stack(all_valid, axis=1)
    valid_pinch = pinch_error[valid_values]
    valid_aperture = aperture_error[valid_values]

    physical = episode.gripper_width_m is not None
    physical_report = None
    if physical:
        requested = np.asarray(episode.gripper_width_m)
        clipped = np.clip(requested, 0, OPENARM_MAX_APERTURE_M)
        physical_error = np.abs(actual_aperture - clipped)
        physical_report = {
            "source_width_unit": "metre",
            "clipped_unreachable_values": int(np.sum(requested > OPENARM_MAX_APERTURE_M)),
            "maximum_supported_aperture_m": OPENARM_MAX_APERTURE_M,
            "aperture_error_m": _summary(physical_error[valid_values]),
        }

    enough_frames = bool(np.any(valid_values))
    pinch_ok = enough_frames and float(np.max(valid_pinch)) <= maximum_pinch_error_m
    aperture_ok = enough_frames and float(np.max(valid_aperture)) <= maximum_aperture_error_m
    return {
        "schema": "openarm-gripper-contact-validation-v1",
        "ok": bool(pinch_ok and aperture_ok),
        "frames": frames,
        "validated_frame_sides": int(np.sum(valid_values)),
        "gripper_semantics": "0=open, 1=closed",
        "target": measurement["target_kind"],
        "pinch_center_compensated": bool(episode.metadata.get("pinch_center_compensated", False)),
        "physical_width_available": physical,
        "thresholds": {
            "maximum_pinch_error_m": maximum_pinch_error_m,
            "maximum_aperture_error_m": maximum_aperture_error_m,
        },
        "pinch_error_m": _summary(valid_pinch),
        "aperture_error_m": _summary(valid_aperture),
        "physical_width": physical_report,
        "sides": side_reports,
    }


def write_gripper_contact_validation(
    episode: Episode, output: str | Path, model_path: str | Path | None = None, **kwargs
) -> Path:
    report = validate_gripper_contact(episode, model_path, **kwargs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output
