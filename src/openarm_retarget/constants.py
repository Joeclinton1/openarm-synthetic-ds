from __future__ import annotations

OPENARM_MUJOCO_REPO = "https://github.com/enactic/openarm_mujoco.git"
OPENARM_MUJOCO_COMMIT = "9eadf86d5b9a0713fdc097019302e02e4b303083"
OPENARM_MODEL_RELATIVE = "v2/openarm_bimanual.xml"

FPS = 30
POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
SIDES = ("right", "left")
JOINT_NAMES = [
    *(f"right_joint{i}.pos" for i in range(1, 8)),
    "right_gripper.pos",
    *(f"left_joint{i}.pos" for i in range(1, 8)),
    "left_gripper.pos",
]

ARM_JOINT_NAMES = {side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in SIDES}
FINGER_JOINT_NAMES = {
    side: [f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"] for side in SIDES
}
EE_SITE_NAMES = {side: f"{side}_ee_control_point" for side in SIDES}
EE_BODY_NAMES = {side: f"openarm_{side}_ee_base_link" for side in SIDES}
