"""Deterministic common wheel motion with learned bounded contact corrections."""

INTERFACE_VERSION = "dexsteer-v2-contact-residual"
FRAME_DIM = 122
ACTION_LAYOUT = ["finger_position_residual_0:12", "common_phase_offset",
                 "differential_phase_offset", "left_radial_offset", "left_axial_offset",
                 "right_radial_offset", "right_axial_offset"]
