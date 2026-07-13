from __future__ import annotations

OPENARM_MUJOCO_REPO = "https://github.com/enactic/openarm_mujoco.git"
OPENARM_MUJOCO_COMMIT = "9eadf86d5b9a0713fdc097019302e02e4b303083"
OPENARM_MODEL_RELATIVE = "v2/openarm_bimanual.xml"

# URLab is intentionally pinned independently of the OpenArm model.  The plugin
# and its Python bridge do not publish binary releases yet, so reproducible
# source revisions are part of every render manifest.
URLAB_REPO = "https://github.com/URLab-Sim/UnrealRoboticsLab.git"
URLAB_COMMIT = "567cbd907a570b820beb87fbddd69c356a6d86da"
URLAB_BRIDGE_REPO = "https://github.com/URLab-Sim/urlab_bridge.git"
# The typed urlab_client used by URLab's current main branch is still on
# upstream's remote-stepping integration branch rather than bridge main.
URLAB_BRIDGE_COMMIT = "bd3a63b15c6430b0b3738a0f5876554d5408644a"
URLAB_MODEL_ID = f"openarm-2.0-bimanual@{OPENARM_MUJOCO_COMMIT}"

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
    side: [f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"]
    for side in SIDES
}
EE_SITE_NAMES = {side: f"{side}_ee_control_point" for side in SIDES}
EE_BODY_NAMES = {side: f"openarm_{side}_ee_base_link" for side in SIDES}
