from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .constants import ARM_JOINT_NAMES, EE_SITE_NAMES, SIDES
from .model import resolve_model
from .poses import normalize_quaternion_xyzw, orientation_error
from .schema import Episode


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 120
    damping: float = 2e-3
    step_size: float = 0.65
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.08
    position_weight: float = 1.0
    orientation_weight: float = 0.35
    posture_weight: float = 0.08
    curvature_weight: float = 0.18
    limit_margin: float = 0.015
    max_step_norm: float = 0.25
    trajectory_curvature_weight: float = 3.0
    failed_frame_data_weight: float = 0.03


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    position_error: float
    orientation_error: float
    iterations: int
    condition: float


class OpenArmIK:
    """Damped least-squares IK with null-space temporal regularization.

    The seventh DoF is resolved toward a constant-velocity seed. That makes the secondary
    objective minimize discrete joint curvature while the primary Cartesian task remains
    unchanged. Joint-centre attraction and bounded updates provide singularity/limit safety.
    """

    def __init__(self, model_path: str | Path | None = None, config: IKConfig | None = None):
        self.model_path = resolve_model(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.config = config or IKConfig()
        self._joints: dict[str, np.ndarray] = {}
        self._dofs: dict[str, np.ndarray] = {}
        self._limits: dict[str, np.ndarray] = {}
        self._sites: dict[str, int] = {}
        for side in SIDES:
            ids = np.array(
                [
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in ARM_JOINT_NAMES[side]
                ]
            )
            if np.any(ids < 0):
                raise ValueError(f"Official model is missing {side} OpenArm joints")
            self._joints[side] = self.model.jnt_qposadr[ids]
            self._dofs[side] = self.model.jnt_dofadr[ids]
            self._limits[side] = self.model.jnt_range[ids].copy()
            self._sites[side] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAMES[side]
            )

    def neutral(self, side: str) -> np.ndarray:
        # Symmetric, flexed, collision-free reference measured against the official v2 model.
        # The exact joint midpoint has a wrist singularity (condition number > 3e5).
        right = np.array([0.61, 1.22, -0.66, 1.64, 0.31, -0.13, 0.10])
        if side == "right":
            return right
        return np.array([-0.61, -1.22, 0.66, 1.64, -0.31, 0.13, -0.10])

    def limits(self, side: str) -> np.ndarray:
        return self._limits[side].copy()

    def set_arm(self, side: str, q: np.ndarray) -> None:
        self.data.qpos[self._joints[side]] = q

    def forward_pose(self, side: str, q: np.ndarray) -> np.ndarray:
        self.set_arm(side, q)
        mujoco.mj_forward(self.model, self.data)
        site = self._sites[side]
        quat_wxyz = np.empty(4)
        mujoco.mju_mat2Quat(quat_wxyz, self.data.site_xmat[site])
        return np.concatenate([self.data.site_xpos[site], quat_wxyz[[1, 2, 3, 0]]])

    def evaluate(
        self, side: str, q: np.ndarray, target_pose: np.ndarray
    ) -> tuple[float, float, float]:
        current = self.forward_pose(side, q)
        position = float(np.linalg.norm(target_pose[:3] - current[:3]))
        orientation = float(np.linalg.norm(orientation_error(target_pose[3:], current[3:])))
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._sites[side])
        jacobian = np.vstack(
            [
                self.config.position_weight * jacp[:, self._dofs[side]],
                self.config.orientation_weight * jacr[:, self._dofs[side]],
            ]
        )
        singular = np.linalg.svd(jacobian, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-12))
        return position, orientation, condition

    def _smooth_trajectory(self, side: str, q: np.ndarray, success: np.ndarray) -> np.ndarray:
        n = len(q)
        weight = self.config.trajectory_curvature_weight
        if n < 3 or weight <= 0:
            return q
        data_weight = np.where(success, 1.0, self.config.failed_frame_data_weight)
        second_difference = diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            [0, 1, 2],
            shape=(n - 2, n),
            format="csc",
        )
        system = diags(data_weight, format="csc") + weight * (
            second_difference.T @ second_difference
        )
        smoothed = spsolve(system, data_weight[:, None] * q)
        limits = self._limits[side]
        return np.clip(
            smoothed,
            limits[:, 0] + self.config.limit_margin,
            limits[:, 1] - self.config.limit_margin,
        )

    def solve(
        self,
        side: str,
        target_pose: np.ndarray,
        seed: np.ndarray | None = None,
        previous: np.ndarray | None = None,
        previous_previous: np.ndarray | None = None,
    ) -> IKResult:
        cfg = self.config
        target_pose = np.asarray(target_pose, dtype=np.float64)
        target_quat = normalize_quaternion_xyzw(target_pose[3:])
        q = np.asarray(seed if seed is not None else self.neutral(side), dtype=np.float64).copy()
        limits = self._limits[side]
        lower = limits[:, 0] + cfg.limit_margin
        upper = limits[:, 1] - cfg.limit_margin
        q = np.clip(q, lower, upper)
        target_posture = self.neutral(side)
        if previous is not None:
            target_posture = np.asarray(previous).copy()
        if previous is not None and previous_previous is not None:
            constant_velocity = 2 * np.asarray(previous) - np.asarray(previous_previous)
            target_posture = (
                cfg.posture_weight * target_posture + cfg.curvature_weight * constant_velocity
            ) / (cfg.posture_weight + cfg.curvature_weight)
        target_posture = np.clip(target_posture, lower, upper)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        condition = np.inf
        pos_norm = ori_norm = np.inf
        for iteration in range(1, cfg.max_iterations + 1):
            self.set_arm(side, q)
            mujoco.mj_forward(self.model, self.data)
            site = self._sites[side]
            current_wxyz = np.empty(4)
            mujoco.mju_mat2Quat(current_wxyz, self.data.site_xmat[site])
            current_xyzw = current_wxyz[[1, 2, 3, 0]]
            pos_error = target_pose[:3] - self.data.site_xpos[site]
            ori_error = orientation_error(target_quat, current_xyzw)
            pos_norm = float(np.linalg.norm(pos_error))
            ori_norm = float(np.linalg.norm(ori_error))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
            dofs = self._dofs[side]
            jac = np.vstack(
                [
                    cfg.position_weight * jacp[:, dofs],
                    cfg.orientation_weight * jacr[:, dofs],
                ]
            )
            singular = np.linalg.svd(jac, compute_uv=False)
            condition = float(singular[0] / max(singular[-1], 1e-12))
            if pos_norm <= cfg.position_tolerance and ori_norm <= cfg.orientation_tolerance:
                return IKResult(q, True, pos_norm, ori_norm, iteration, condition)

            error = np.concatenate(
                [
                    cfg.position_weight * pos_error,
                    cfg.orientation_weight * ori_error,
                ]
            )
            jjt = jac @ jac.T
            inverse = np.linalg.solve(jjt + cfg.damping**2 * np.eye(6), np.eye(6))
            pinv = jac.T @ inverse
            dq_primary = pinv @ error
            null = np.eye(7) - pinv @ jac
            ranges = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)
            centre = np.mean(limits, axis=1)
            posture_gradient = (target_posture - q) / ranges
            limit_gradient = -0.02 * (q - centre) / ranges
            dq = dq_primary + null @ (posture_gradient + limit_gradient)
            norm = np.linalg.norm(dq)
            if norm > cfg.max_step_norm:
                dq *= cfg.max_step_norm / norm
            q = np.clip(q + cfg.step_size * dq, lower, upper)

        return IKResult(q, False, pos_norm, ori_norm, cfg.max_iterations, condition)

    def solve_episode(self, episode: Episode) -> Episode:
        episode.validate()
        n = len(episode.timestamp)
        joints = np.zeros((n, 2, 7), dtype=np.float64)
        success = np.zeros((n, 2), dtype=bool)
        raw_success_all = np.zeros((n, 2), dtype=bool)
        position_error = np.full((n, 2), np.inf)
        orientation_error_values = np.full((n, 2), np.inf)
        condition = np.full((n, 2), np.inf)
        active_sides = set(episode.metadata.get("active_sides", SIDES))
        for side_index, side in enumerate(SIDES):
            if side not in active_sides:
                joints[:, side_index] = self.neutral(side)
                success[:, side_index] = True
                position_error[:, side_index] = 0
                orientation_error_values[:, side_index] = 0
                condition[:, side_index] = 1
                raw_success_all[:, side_index] = True
                continue
            q_prev = self.neutral(side)
            q_prev2: np.ndarray | None = None
            for frame in range(n):
                candidates = [
                    self.solve(
                        side,
                        episode.ee_pose[frame, side_index],
                        seed=q_prev,
                        previous=q_prev,
                        previous_previous=q_prev2,
                    )
                ]
                if not candidates[0].success:
                    reference = self.neutral(side)
                    candidates.append(
                        self.solve(
                            side,
                            episode.ee_pose[frame, side_index],
                            seed=reference,
                            previous=reference,
                        )
                    )
                    for elbow_delta in (-0.45, 0.45):
                        perturbed = reference.copy()
                        perturbed[3] = np.clip(
                            perturbed[3] + elbow_delta,
                            self._limits[side][3, 0] + self.config.limit_margin,
                            self._limits[side][3, 1] - self.config.limit_margin,
                        )
                        candidates.append(
                            self.solve(
                                side,
                                episode.ee_pose[frame, side_index],
                                seed=perturbed,
                                previous=q_prev,
                                previous_previous=q_prev2,
                            )
                        )
                result = min(
                    candidates,
                    key=lambda value: (
                        not value.success,
                        value.position_error / self.config.position_tolerance
                        + value.orientation_error / self.config.orientation_tolerance,
                        np.linalg.norm(value.q - q_prev),
                    ),
                )
                joints[frame, side_index] = result.q
                success[frame, side_index] = result.success
                position_error[frame, side_index] = result.position_error
                orientation_error_values[frame, side_index] = result.orientation_error
                condition[frame, side_index] = result.condition
                q_prev2, q_prev = q_prev, result.q
            raw_success = success[:, side_index].copy()
            raw_success_all[:, side_index] = raw_success
            joints[:, side_index] = self._smooth_trajectory(
                side, joints[:, side_index], raw_success
            )
            for frame in range(n):
                position, orientation, jacobian_condition = self.evaluate(
                    side, joints[frame, side_index], episode.ee_pose[frame, side_index]
                )
                position_error[frame, side_index] = position
                orientation_error_values[frame, side_index] = orientation
                condition[frame, side_index] = jacobian_condition
                success[frame, side_index] = (
                    position <= self.config.position_tolerance
                    and orientation <= self.config.orientation_tolerance
                )
        episode.joint_position = joints
        episode.diagnostics.update(
            {
                "ik_raw_success": raw_success_all,
                "ik_success": success,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error_values,
                "jacobian_condition": condition,
            }
        )
        return episode

    def arm_collisions(
        self, right_q: np.ndarray, left_q: np.ndarray, include_self: bool = True
    ) -> list[tuple[str, str]]:
        self.set_arm("right", right_q)
        self.set_arm("left", left_q)
        mujoco.mj_forward(self.model, self.data)
        contacts: list[tuple[str, str]] = []
        for contact in self.data.contact:
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            # The two coupled finger meshes touch near zero opening by design.
            if "finger" in name1 and "finger" in name2:
                continue
            interarm = ("left" in name1 and "right" in name2) or (
                "right" in name1 and "left" in name2
            )
            same_arm = include_self and any(side in name1 and side in name2 for side in SIDES)
            if interarm or same_arm:
                contacts.append((name1, name2))
        return contacts
