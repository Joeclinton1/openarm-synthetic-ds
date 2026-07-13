from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import SIDES
from .gripper_validation import measure_gripper_contact
from .ik import OpenArmIK
from .schema import Episode


@dataclass(frozen=True)
class FilterConfig:
    max_position_error_m: float = 0.01
    max_orientation_error_rad: float = 0.15
    max_joint_velocity_rad_s: float = 3.0
    max_joint_acceleration_rad_s2: float = 20.0
    max_jacobian_condition: float = 500.0
    minimum_joint_limit_margin_rad: float = 0.0
    reject_interarm_collision: bool = True
    reject_self_collision: bool = True
    minimum_run_frames: int = 6
    max_pinch_center_error_m: float = 0.01
    max_gripper_aperture_error_m: float = 1e-6


def _remove_short_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = mask.copy()
    start = 0
    while start < len(result):
        end = start + 1
        while end < len(result) and result[end] == result[start]:
            end += 1
        if result[start] and end - start < minimum:
            result[start:end] = False
        start = end
    return result


def filter_episode(episode: Episode, ik: OpenArmIK, config: FilterConfig | None = None) -> Episode:
    cfg = config or FilterConfig()
    if episode.joint_position is None:
        raise ValueError("Run IK before feasibility filtering")
    n = len(episode.timestamp)
    feasible = np.ones(n, dtype=bool)
    diagnostics = episode.diagnostics
    active = [SIDES.index(side) for side in episode.metadata.get("active_sides", SIDES)]
    # Solver convergence is retained as a diagnostic, while acceptance is based on the
    # recomputed Cartesian residual below. A smoothed trajectory can be valid even where the
    # original frame-wise iteration exhausted its budget.
    feasible &= np.all(
        diagnostics["position_error_m"][:, active] <= cfg.max_position_error_m, axis=1
    )
    feasible &= np.all(
        diagnostics["orientation_error_rad"][:, active] <= cfg.max_orientation_error_rad, axis=1
    )
    feasible &= np.all(
        diagnostics["jacobian_condition"][:, active] <= cfg.max_jacobian_condition, axis=1
    )

    dt = np.diff(episode.timestamp)
    velocity = np.zeros_like(episode.joint_position)
    acceleration = np.zeros_like(episode.joint_position)
    if n > 1:
        velocity[1:] = np.diff(episode.joint_position, axis=0) / dt[:, None, None]
        feasible[1:] &= (
            np.max(np.abs(velocity[1:, active]), axis=(1, 2)) <= cfg.max_joint_velocity_rad_s
        )
    if n > 2:
        dt_acc = ((dt[1:] + dt[:-1]) / 2)[:, None, None]
        acceleration[2:] = np.diff(velocity[1:], axis=0) / dt_acc
        feasible[2:] &= (
            np.max(np.abs(acceleration[2:, active]), axis=(1, 2))
            <= cfg.max_joint_acceleration_rad_s2
        )

    joint_limit_violation = np.zeros(n, dtype=bool)
    for side_index in active:
        limits = ik.limits(SIDES[side_index])
        q = episode.joint_position[:, side_index]
        joint_limit_violation |= np.any(
            (q < limits[:, 0] + cfg.minimum_joint_limit_margin_rad)
            | (q > limits[:, 1] - cfg.minimum_joint_limit_margin_rad),
            axis=1,
        )
    feasible &= ~joint_limit_violation

    contact = measure_gripper_contact(episode, ik.model_path)
    pinch_error = np.asarray(contact["pinch_error_m"])
    aperture_error = np.asarray(contact["aperture_error_m"])
    feasible &= np.all(pinch_error[:, active] <= cfg.max_pinch_center_error_m, axis=1)
    feasible &= np.all(aperture_error[:, active] <= cfg.max_gripper_aperture_error_m, axis=1)

    collision = np.zeros(n, dtype=bool)
    if cfg.reject_interarm_collision or cfg.reject_self_collision:
        for frame in range(n):
            collision[frame] = bool(
                ik.arm_collisions(
                    episode.joint_position[frame, 0],
                    episode.joint_position[frame, 1],
                    include_self=cfg.reject_self_collision,
                    gripper=episode.gripper[frame],
                )
            )
        feasible &= ~collision
    feasible = _remove_short_runs(feasible, cfg.minimum_run_frames)
    episode.feasible = feasible
    episode.diagnostics.update(
        {
            "joint_velocity_rad_s": velocity,
            "joint_acceleration_rad_s2": acceleration,
            "invalid_joint_limit": joint_limit_violation,
            "invalid_collision": collision,
            "pinch_center_error_m": pinch_error,
            "gripper_aperture_error_m": aperture_error,
        }
    )
    return episode
