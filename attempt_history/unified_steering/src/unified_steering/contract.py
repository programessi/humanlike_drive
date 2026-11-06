"""Versioned observation/action layout, implemented by observations/execution."""

INTERFACE_VERSION = "unified-steering-v1"
OBSERVATION_FIELDS = (
    ("arm_position", 14),
    ("arm_velocity", 14),
    ("arm_target_error", 14),
    ("hand_position", 12),
    ("hand_velocity", 12),
    ("hand_target_error", 12),
    ("wrist_position", 6),
    ("wrist_rotation_6d", 12),
    ("wrist_twist", 12),
    ("previous_executed_action", 26),
    ("wheel_state", 4),
    ("reference_error_preview", 11),
    ("reference_velocity", 1),
    ("wheel_geometry", 2),
)
OBSERVATION_DIM = sum(size for _, size in OBSERVATION_FIELDS)
ACTION_DIM = 26
PRIVILEGED_DIM = 20
CRITIC_DIM = OBSERVATION_DIM + PRIVILEGED_DIM
PREVIEW_INDICES = slice(138, 150)

ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_SUFFIXES = ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
                 "middle_proximal", "ring_proximal", "pinky_proximal")
ARM_NAMES = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in ARM_SUFFIXES)
HAND_NAMES = tuple(f"{side}_{joint}_joint" for side in ("L", "R") for joint in HAND_SUFFIXES)
MOTOR_NAMES = ARM_NAMES + HAND_NAMES
