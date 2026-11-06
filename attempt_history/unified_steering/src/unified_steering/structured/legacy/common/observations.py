"""Shared actor observation contract; no contacts, loads or privileged physics."""
from __future__ import annotations

import numpy as np

FRAME_DIM = 120
PRIVILEGED_DIM = 54


def actor_frame(q, estimated_velocity, motor_target, previous_action, wrist_position,
                wrist_rotation, wheel_angle, wheel_velocity, target_angle,
                target_velocity, rim_radius):
    # Rotations use the first two columns, avoiding Euler-angle discontinuities.
    wrist = np.concatenate([np.asarray(wrist_position) / .2,
                            np.asarray(wrist_rotation)[:, :, :2].reshape(2, 6)], axis=1)
    values = np.r_[q, np.asarray(estimated_velocity) / 5,
                   np.asarray(motor_target) - q, previous_action, wrist.ravel(),
                   wheel_angle, wheel_velocity, target_angle, target_velocity,
                   target_angle - wheel_angle, rim_radius / .14]
    if values.shape != (FRAME_DIM,) or not np.isfinite(values).all():
        raise ValueError("Invalid actor sensor frame")
    return values.astype(np.float32)

