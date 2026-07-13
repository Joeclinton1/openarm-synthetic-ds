from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .constants import ARM_JOINT_NAMES, EE_BODY_NAMES, FINGER_JOINT_NAMES, SIDES
from .model import resolve_model
from .poses import matrix_to_pose, pose_to_matrix

# Geometry measured from the pinned official OpenArm 2.0 MJCF.  The pad point is
# the centre of the parallel contact face at the distal end of each finger.
OPENARM_FINGER_MAX_ANGLE_RAD = 0.7854
OPENARM_FINGER_PIVOT_X_M = -0.00143
OPENARM_FINGER_PIVOT_OFFSET_M = 0.018
OPENARM_FINGER_PIVOT_Z_M = -0.068
OPENARM_FINGER_PAD_REACH_M = 0.09534


def opening_angle_to_aperture(angle_rad: np.ndarray | float) -> np.ndarray:
    """Return pad-centre separation in metres for an opening angle."""
    angle = np.clip(np.asarray(angle_rad, dtype=np.float64), 0, OPENARM_FINGER_MAX_ANGLE_RAD)
    pivot = OPENARM_FINGER_PIVOT_OFFSET_M
    reach = OPENARM_FINGER_PAD_REACH_M
    return 2 * (pivot - pivot * np.cos(angle) + reach * np.sin(angle))


OPENARM_MAX_APERTURE_M = float(opening_angle_to_aperture(OPENARM_FINGER_MAX_ANGLE_RAD))
OPENARM_CLOSED_PINCH_MIDPOINT_M = np.array(
    [
        OPENARM_FINGER_PIVOT_X_M,
        0.0,
        OPENARM_FINGER_PIVOT_Z_M - OPENARM_FINGER_PAD_REACH_M,
    ],
    dtype=np.float64,
)


def aperture_to_opening_angle(aperture_m: np.ndarray | float) -> np.ndarray:
    """Invert the official finger geometry, clipping unreachable apertures."""
    aperture = np.clip(np.asarray(aperture_m, dtype=np.float64), 0, OPENARM_MAX_APERTURE_M)
    pivot = OPENARM_FINGER_PIVOT_OFFSET_M
    reach = OPENARM_FINGER_PAD_REACH_M
    radius = np.hypot(pivot, reach)
    phase = np.arctan2(pivot, reach)
    angle = np.arcsin(np.clip((aperture / 2 - pivot) / radius, -1, 1)) + phase
    return np.clip(angle, 0, OPENARM_FINGER_MAX_ANGLE_RAD)


def aperture_to_closure(aperture_m: np.ndarray | float) -> np.ndarray:
    """Convert physical pad separation to the canonical 0=open, 1=closed value."""
    return 1 - aperture_to_opening_angle(aperture_m) / OPENARM_FINGER_MAX_ANGLE_RAD


def closure_to_aperture(closure: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(closure, dtype=np.float64), 0, 1)
    return opening_angle_to_aperture((1 - values) * OPENARM_FINGER_MAX_ANGLE_RAD)


def closure_to_finger_qpos(gripper: np.ndarray) -> np.ndarray:
    """Expand canonical [..., right, left] closure to official four-joint order."""
    values = np.asarray(gripper, dtype=np.float64)
    if values.shape[-1:] != (2,):
        raise ValueError(f"gripper must end in right/left dimension 2, got {values.shape}")
    opening = (1 - np.clip(values, 0, 1)) * OPENARM_FINGER_MAX_ANGLE_RAD
    return np.stack([-opening[..., 0], -opening[..., 0], opening[..., 1], opening[..., 1]], axis=-1)


def pinch_midpoint_local(closure: np.ndarray | float) -> np.ndarray:
    """Pinch midpoint in the OpenArm EE-base frame for each closure value."""
    values = np.clip(np.asarray(closure, dtype=np.float64), 0, 1)
    angle = (1 - values) * OPENARM_FINGER_MAX_ANGLE_RAD
    z = OPENARM_FINGER_PIVOT_Z_M - (
        OPENARM_FINGER_PIVOT_OFFSET_M * np.sin(angle) + OPENARM_FINGER_PAD_REACH_M * np.cos(angle)
    )
    return np.stack([np.full_like(z, OPENARM_FINGER_PIVOT_X_M), np.zeros_like(z), z], axis=-1)


def preserve_pinch_center(
    ee_pose: np.ndarray,
    gripper: np.ndarray,
    target_offset_local_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the EE base so the moving-jaw midpoint stays at the closed-pad target.

    Returns the compensated EE-base poses and the invariant world-frame pinch target.
    The input pose is the closed-gripper registration result.
    """
    pose = np.asarray(ee_pose, dtype=np.float64)
    closure = np.asarray(gripper, dtype=np.float64)
    if pose.shape[:-1] != closure.shape or pose.shape[-1] != 7:
        raise ValueError("ee_pose and gripper leading dimensions must match")
    transforms = pose_to_matrix(pose)
    rotation = transforms[..., :3, :3]
    closed = OPENARM_CLOSED_PINCH_MIDPOINT_M
    target_local = np.broadcast_to(closed, (*closure.shape, 3)).copy()
    if target_offset_local_m is not None:
        offset = np.asarray(target_offset_local_m, dtype=np.float64)
        if offset.shape != target_local.shape:
            raise ValueError("target_offset_local_m must have shape gripper.shape + (3,)")
        target_local += offset
    target = transforms[..., :3, 3] + np.einsum("...ij,...j->...i", rotation, target_local)
    current = pinch_midpoint_local(closure)
    transforms[..., :3, 3] += np.einsum(
        "...ij,...j->...i", rotation, target_local - current
    )
    return matrix_to_pose(transforms), target


def finger_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for side in SIDES
        for name in FINGER_JOINT_NAMES[side]
    ]
    if any(joint_id < 0 for joint_id in ids):
        raise ValueError("Official model is missing OpenArm finger joints")
    return model.jnt_qposadr[np.asarray(ids)]


def set_gripper_qpos(model: mujoco.MjModel, data: mujoco.MjData, gripper: np.ndarray) -> None:
    data.qpos[finger_qpos_addresses(model)] = closure_to_finger_qpos(gripper)


class OpenArmPinchKinematics:
    """MuJoCo-backed fingertip-pad measurement used by production validation."""

    _PAD_LOCAL = {
        "right": (
            np.array([0.0, -OPENARM_FINGER_PIVOT_OFFSET_M, -OPENARM_FINGER_PAD_REACH_M]),
            np.array([0.0, OPENARM_FINGER_PIVOT_OFFSET_M, -OPENARM_FINGER_PAD_REACH_M]),
        ),
        "left": (
            np.array([0.0, OPENARM_FINGER_PIVOT_OFFSET_M, -OPENARM_FINGER_PAD_REACH_M]),
            np.array([0.0, -OPENARM_FINGER_PIVOT_OFFSET_M, -OPENARM_FINGER_PAD_REACH_M]),
        ),
    }

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = resolve_model(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.arm_qpos: dict[str, np.ndarray] = {}
        self.finger_bodies: dict[str, tuple[int, int]] = {}
        self.ee_bodies: dict[str, int] = {}
        for side in SIDES:
            arm_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES[side]
            ]
            finger_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in FINGER_JOINT_NAMES[side]
            ]
            self.arm_qpos[side] = self.model.jnt_qposadr[np.asarray(arm_ids)]
            self.finger_bodies[side] = tuple(int(self.model.jnt_bodyid[j]) for j in finger_ids)
            self.ee_bodies[side] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAMES[side]
            )
        self.finger_qpos = finger_qpos_addresses(self.model)

    def set_frame(self, joint_position: np.ndarray, gripper: np.ndarray) -> None:
        joints = np.asarray(joint_position, dtype=np.float64)
        if joints.shape != (2, 7):
            raise ValueError("joint_position frame must have shape [2,7]")
        for index, side in enumerate(SIDES):
            self.data.qpos[self.arm_qpos[side]] = joints[index]
        self.data.qpos[self.finger_qpos] = closure_to_finger_qpos(gripper)
        mujoco.mj_forward(self.model, self.data)

    def pad_points(self, side: str) -> np.ndarray:
        points = []
        for body, local in zip(self.finger_bodies[side], self._PAD_LOCAL[side], strict=True):
            rotation = self.data.xmat[body].reshape(3, 3)
            points.append(self.data.xpos[body] + rotation @ local)
        return np.stack(points)

    def measure(self, side: str) -> tuple[np.ndarray, float]:
        pads = self.pad_points(side)
        return np.mean(pads, axis=0), float(np.linalg.norm(pads[0] - pads[1]))
