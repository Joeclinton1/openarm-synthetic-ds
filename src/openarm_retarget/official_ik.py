from __future__ import annotations

import contextlib
import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import (
    DORA_OPENARM_KINEMATICS_COMMIT,
    DORA_OPENARM_KINEMATICS_REPO,
    OPENARM_CONTROL_COMMIT,
    OPENARM_CONTROL_REPO,
    SIDES,
)
from .ik import OpenArmIK
from .model import resolve_model
from .schema import Episode


@dataclass(frozen=True)
class OfficialIKConfig:
    """Published OpenArm MuJoCo/Mink settings adapted for offline trajectories.

    The upstream Dora node defaults to five QP iterations per incoming control
    event. Offline conversion uses more iterations because every recorded frame
    must independently meet the Cartesian gate; it still calls the unchanged
    upstream Kinematics implementation with its published costs and solver.
    """

    max_iterations: int = 80
    dt: float = 0.1
    damping: float = 0.25
    lm_damping: float = 0.01
    posture_cost: float = 0.01
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    solver: str = "daqp"
    max_target_retries: int = 2
    smooth_trajectory: bool = False


class OfficialOpenArmIK:
    """Adapter around the IK implementation used by dora-openarm-kinematics."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: OfficialIKConfig | None = None,
    ) -> None:
        try:
            from openarm_control import ArmSetup, IKParams, Kinematics
        except ImportError as exc:
            raise RuntimeError(
                "Official OpenArm IK dependencies are missing; run "
                "`uv sync --extra official-ik --extra dev`"
            ) from exc

        self.model_path = resolve_model(model_path)
        self.config = config or OfficialIKConfig()
        self._ArmSetup = ArmSetup
        self._IKParams = IKParams
        self._Kinematics = Kinematics
        # Reuse the audited evaluator, collision checker, limits, neutral seed,
        # and production trajectory smoother for an apples-to-apples comparison.
        self.evaluator = OpenArmIK(self.model_path)
        self.solve_seconds = 0.0

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "interface": {
                "repository": DORA_OPENARM_KINEMATICS_REPO,
                "commit": DORA_OPENARM_KINEMATICS_COMMIT,
                "release": "0.1.1",
            },
            "implementation": {
                "repository": OPENARM_CONTROL_REPO,
                "commit": OPENARM_CONTROL_COMMIT,
                "release": "0.1.0",
            },
            "config": asdict(self.config),
            "adapter": "direct Kinematics API; equivalent solver path without Dora transport",
            "initialization": "existing collision-free OpenArm neutral posture",
            "upstream_limit_fallback": (
                "openarm-control 0.1.0 retries a failed constrained QP with limits=[]"
            ),
        }

    @staticmethod
    def _official_pose(pose_xyzw: np.ndarray) -> np.ndarray:
        """Convert canonical xyz+xyzw to the official xyz+wxyz convention."""
        return np.concatenate([pose_xyzw[:3], pose_xyzw[[6, 3, 4, 5]]]).astype(np.float32)

    def _make_kinematics(self, active_sides: tuple[str, ...]):
        if active_sides == SIDES:
            mode = "bimanual"
        elif active_sides == ("right",):
            mode = "right"
        elif active_sides == ("left",):
            mode = "left"
        else:
            raise ValueError(f"Unsupported active_sides: {active_sides}")
        setup = self._ArmSetup.from_args(
            str(self.model_path),
            mode,
            "right_ee_control_point",
            "site",
            "left_ee_control_point",
            "site",
            None,
        )
        cfg = self.config
        params = self._IKParams(
            position_cost=cfg.position_cost,
            orientation_cost=cfg.orientation_cost,
            lm_damping=cfg.lm_damping,
            damping=cfg.damping,
            solver=cfg.solver,
            posture_cost=cfg.posture_cost,
            dt=cfg.dt,
            max_iters=cfg.max_iterations,
        )
        # openarm-control 0.1.0 prints internal qpos/dof sets during construction.
        # Suppress that upstream debug output in normal batch conversion.
        with contextlib.redirect_stdout(io.StringIO()):
            return self._Kinematics(setup, params)

    def solve_episode(self, episode: Episode) -> Episode:
        episode.validate()
        n = len(episode.timestamp)
        active_sides = tuple(
            side for side in SIDES if side in episode.metadata.get("active_sides", SIDES)
        )
        if not active_sides:
            raise ValueError("Episode must have at least one active arm")
        kin = self._make_kinematics(active_sides)

        seed = np.concatenate(
            [
                self.evaluator.neutral("right"),
                [float(episode.gripper[0, 0]) if n else 0.0],
                self.evaluator.neutral("left"),
                [float(episode.gripper[0, 1]) if n else 0.0],
            ]
        ).astype(np.float32)
        kin.sync(seed)

        joints = np.zeros((n, 2, 7), dtype=np.float64)
        joints[:, 0] = self.evaluator.neutral("right")
        joints[:, 1] = self.evaluator.neutral("left")
        solver_failed = np.zeros(n, dtype=bool)
        target_retries = np.zeros(n, dtype=np.int32)
        started = time.perf_counter()
        for frame in range(n):
            result = None
            for attempt in range(self.config.max_target_retries + 1):
                for side_index, side in enumerate(SIDES):
                    if side not in active_sides:
                        continue
                    kin.set_target(side, self._official_pose(episode.ee_pose[frame, side_index]))
                    kin.set_gripper(side, float(episode.gripper[frame, side_index]))
                candidate = kin.solve()
                if (
                    candidate is None
                    or candidate.shape != (16,)
                    or not np.all(np.isfinite(candidate))
                ):
                    continue
                result = candidate
                target_retries[frame] = attempt
                candidate_joints = (candidate[:7], candidate[8:15])
                converged = True
                for side_index, side in enumerate(SIDES):
                    if side not in active_sides:
                        continue
                    position, orientation, _ = self.evaluator.evaluate(
                        side, candidate_joints[side_index], episode.ee_pose[frame, side_index]
                    )
                    converged &= (
                        position <= self.evaluator.config.position_tolerance
                        and orientation <= self.evaluator.config.orientation_tolerance
                    )
                if converged:
                    break
            if result is None:
                solver_failed[frame] = True
                target_retries[frame] = self.config.max_target_retries
                if frame:
                    joints[frame] = joints[frame - 1]
                continue
            joints[frame, 0] = result[:7]
            joints[frame, 1] = result[8:15]
        self.solve_seconds = time.perf_counter() - started

        raw_success = np.zeros((n, 2), dtype=bool)
        for side_index, side in enumerate(SIDES):
            if side not in active_sides:
                raw_success[:, side_index] = True
                continue
            for frame in range(n):
                position, orientation, _ = self.evaluator.evaluate(
                    side, joints[frame, side_index], episode.ee_pose[frame, side_index]
                )
                raw_success[frame, side_index] = (
                    position <= self.evaluator.config.position_tolerance
                    and orientation <= self.evaluator.config.orientation_tolerance
                    and not solver_failed[frame]
                )
            if self.config.smooth_trajectory:
                joints[:, side_index] = self.evaluator._smooth_trajectory(
                    side, joints[:, side_index], raw_success[:, side_index]
                )

        position_error = np.zeros((n, 2), dtype=np.float64)
        orientation_error = np.zeros((n, 2), dtype=np.float64)
        condition = np.ones((n, 2), dtype=np.float64)
        success = np.ones((n, 2), dtype=bool)
        for side_index, side in enumerate(SIDES):
            if side not in active_sides:
                continue
            for frame in range(n):
                position, orientation, jacobian_condition = self.evaluator.evaluate(
                    side, joints[frame, side_index], episode.ee_pose[frame, side_index]
                )
                position_error[frame, side_index] = position
                orientation_error[frame, side_index] = orientation
                condition[frame, side_index] = jacobian_condition
                success[frame, side_index] = (
                    position <= self.evaluator.config.position_tolerance
                    and orientation <= self.evaluator.config.orientation_tolerance
                    and not solver_failed[frame]
                )

        episode.joint_position = joints
        episode.diagnostics.update(
            {
                "ik_raw_success": raw_success,
                "ik_success": success,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "jacobian_condition": condition,
                "official_solver_failed": solver_failed,
                "official_target_retries": target_retries,
            }
        )
        episode.metadata["ik_solver"] = self.provenance
        return episode
