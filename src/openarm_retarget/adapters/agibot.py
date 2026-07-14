from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..gripper import aperture_to_closure, closure_to_aperture, preserve_pinch_center
from ..poses import pose_to_matrix
from ..schema import Episode, SourceConfig


def _interpolate_held_gripper_state(
    values: np.ndarray,
    timestamp: np.ndarray,
    max_gap_s: float,
    transition_duration_s: float,
) -> np.ndarray:
    """Reconstruct continuous motion envelopes from delayed zero-order-held state.

    Each cluster identifies a real open/close event and its settled endpoint. Motion starts
    at the first measured change and follows a fixed-duration smoothstep to that endpoint.
    Intermediate held amplitudes are retained in diagnostics, but are not treated as
    frame-synchronous jaw widths because AgiBot records them well after the visible motion.
    """
    if max_gap_s <= 0:
        raise ValueError("gripper_interpolation_max_gap_s must be positive")
    if transition_duration_s <= 0:
        raise ValueError("gripper_transition_duration_s must be positive")
    state = np.asarray(values, dtype=np.float64)
    time = np.asarray(timestamp, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] != len(time):
        raise ValueError("gripper state must have shape [frames, grippers]")
    result = state.copy()
    for column in range(state.shape[1]):
        signal = state[:, column]
        changes = np.flatnonzero(np.abs(np.diff(signal)) > 1e-12) + 1
        if not len(changes):
            continue
        boundaries = np.flatnonzero(np.diff(time[changes]) > max_gap_s) + 1
        for group in np.split(changes, boundaries):
            anchor = max(int(group[0]) - 1, 0)
            region = np.arange(anchor, int(group[-1]) + 1)
            phase = np.clip(
                (time[region] - time[anchor]) / transition_duration_s, 0.0, 1.0
            )
            smoothstep = phase * phase * (3.0 - 2.0 * phase)
            result[region, column] = signal[anchor] + (
                signal[int(group[-1])] - signal[anchor]
            ) * smoothstep
    return result


def load_agibot_h5(
    path: str | Path,
    config: SourceConfig,
    allow_uncalibrated: bool = False,
) -> Episode:
    if not config.calibrated and not allow_uncalibrated:
        raise ValueError("AgiBot base/flange calibration has not been validated")
    with h5py.File(path, "r") as data:
        raw_timestamp = np.asarray(data[config.fields.get("timestamp", "/timestamp")])
        # Subtract in integer nanoseconds before conversion. Casting epoch-scale nanoseconds
        # directly to float64 needlessly discards timing precision.
        timestamp = (raw_timestamp - raw_timestamp[0]).astype(np.float64) * 1e-9
        position = np.asarray(data[config.fields.get("position", "/state/end/position")])
        orientation = np.asarray(data[config.fields.get("orientation", "/state/end/orientation")])
        raw_gripper = np.asarray(data[config.fields.get("gripper", "/state/effector/position")])
    # AgiBot publishes left then right; canonical storage is right then left.
    raw_pose = np.concatenate([position, orientation], axis=-1)[:, [1, 0]]
    pose = np.stack(
        [
            config.pose_transform(side).apply(raw_pose[:, side_index])
            for side_index, side in enumerate(("right", "left"))
        ],
        axis=1,
    )
    gripper = raw_gripper[:, [1, 0]].astype(np.float64)
    gripper_width_m = None
    if config.gripper_mode == "width_mm":
        gripper_width_m = gripper * 1e-3
        gripper = aperture_to_closure(gripper_width_m)
    elif config.gripper_mode == "closure_position":
        if config.gripper_open_value is None or config.gripper_closed_value is None:
            raise ValueError("closure_position requires gripper_open_value/gripper_closed_value")
        span = config.gripper_closed_value - config.gripper_open_value
        if abs(span) < 1e-9:
            raise ValueError("gripper open and closed values must differ")
        gripper = np.clip((gripper - config.gripper_open_value) / span, 0, 1)
        raw_normalized_gripper = gripper.copy()
        if config.gripper_interpolation == "held_state_smoothstep":
            gripper = _interpolate_held_gripper_state(
                gripper,
                timestamp,
                config.gripper_interpolation_max_gap_s,
                config.gripper_transition_duration_s,
            )
        elif config.gripper_interpolation != "none":
            raise ValueError(
                f"Unsupported gripper interpolation: {config.gripper_interpolation}"
            )
        gripper_width_m = closure_to_aperture(gripper)
    elif np.nanmax(gripper) > 1:
        gripper /= max(float(np.nanpercentile(gripper, 99)), 1e-6)
    diagnostics = {}
    if config.gripper_mode == "closure_position":
        diagnostics["gripper_raw_normalized"] = raw_normalized_gripper
    if config.preserve_pinch_center:
        offset = _source_pinch_offset_in_openarm_frame(config, gripper)
        pose, diagnostics["pinch_center_target_m"] = preserve_pinch_center(pose, gripper, offset)
    episode = Episode(
        timestamp=timestamp,
        ee_pose=pose,
        gripper=np.clip(gripper, 0, 1),
        task=config.name,
        source_dataset=config.repo_id,
        source_episode=Path(path).parent.name,
        gripper_width_m=gripper_width_m,
        diagnostics=diagnostics,
        metadata={
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "active_sides": ["right", "left"],
            "pinch_center_compensated": config.preserve_pinch_center,
            "gripper_calibration": "physical_width_m"
            if gripper_width_m is not None
            else "normalized_only",
            "gripper_interpolation": config.gripper_interpolation,
            "gripper_interpolation_max_gap_s": config.gripper_interpolation_max_gap_s,
            "gripper_transition_duration_s": config.gripper_transition_duration_s,
        },
    )
    episode.validate()
    return episode


def _source_pinch_offset_in_openarm_frame(
    config: SourceConfig, gripper: np.ndarray
) -> np.ndarray | None:
    """Map source jaw midpoint travel into each OpenArm flange frame."""
    if config.source_pinch_center_open_m is None or config.source_pinch_center_closed_m is None:
        return None

    def point(value: dict[str, list[float]] | list[float], side: str) -> np.ndarray:
        selected = value[side] if isinstance(value, dict) else value
        result = np.asarray(selected, dtype=np.float64)
        if result.shape != (3,):
            raise ValueError("source pinch centers must contain xyz triples")
        return result

    result = np.empty((*gripper.shape, 3), dtype=np.float64)
    for side_index, side in enumerate(("right", "left")):
        source_open = point(config.source_pinch_center_open_m, side)
        source_closed = point(config.source_pinch_center_closed_m, side)
        source_delta = (1 - gripper[:, side_index, None]) * (source_open - source_closed)
        source_from_openarm = pose_to_matrix(
            config.pose_transform(side).source_tool_from_openarm_tool
        )
        result[:, side_index] = source_delta @ source_from_openarm[:3, :3]
    return result


def load_agibot_lerobot_episode(
    path: str | Path | pa.Table,
    config: SourceConfig,
    episode_index: int,
    allow_uncalibrated: bool = False,
) -> Episode:
    """Load the provenance-labelled GT-111 AgiBot EE conversion."""
    if not config.calibrated and not allow_uncalibrated:
        raise ValueError("AgiBot base/flange calibration has not been validated")
    table = path if isinstance(path, pa.Table) else pq.read_table(path)
    mask = np.asarray(table["episode_index"]) == episode_index
    if not np.any(mask):
        raise KeyError(episode_index)
    timestamp = np.asarray(table["timestamp"])[mask].astype(np.float64)
    state = np.asarray(table[config.fields.get("pose", "observation.state")].to_pylist())[mask]
    pose = np.empty((len(timestamp), 2, 7), dtype=np.float64)
    gripper = np.empty((len(timestamp), 2), dtype=np.float64)
    # Source state is left xyz-wxyz-gripper, then right; canonical order is right, left.
    for source_index, target_index in ((0, 1), (1, 0)):
        offset = source_index * 8
        raw = np.concatenate(
            [
                state[:, offset : offset + 3],
                state[:, [offset + 4, offset + 5, offset + 6, offset + 3]],
            ],
            axis=1,
        )
        pose[:, target_index] = config.pose_transform(
            "left" if source_index == 0 else "right"
        ).apply(raw)
        gripper[:, target_index] = np.clip(state[:, offset + 7], 0, 1)
    diagnostics = {}
    if config.preserve_pinch_center:
        pose, diagnostics["pinch_center_target_m"] = preserve_pinch_center(pose, gripper)
    result = Episode(
        timestamp=timestamp,
        ee_pose=pose,
        gripper=gripper,
        task="AgiBot World Alpha derived fallback",
        source_dataset=config.repo_id,
        source_episode=str(episode_index),
        diagnostics=diagnostics,
        metadata={
            "calibrated": config.calibrated,
            "source_config": config.to_json(),
            "active_sides": ["right", "left"],
            "provenance_warning": "accessible third-party conversion, not gated original",
            "pinch_center_compensated": config.preserve_pinch_center,
            "gripper_calibration": "normalized_only",
        },
    )
    result.validate()
    return result
