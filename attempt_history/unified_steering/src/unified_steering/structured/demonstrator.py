"""Explicit scripted teacher for feasibility and training labels only.

This object is never installed in the learned execution controller. It emits
the same 24 actions, subject to identical filters, QP and actuator limits.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class DemonstratorConfig:
    angle_gain: float = 2.
    preview_seconds: float = .15
    position_gain: float = 3.
    orientation_gain: float = 3.
    hand_gain: float = 1.


class Demonstrator:
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or DemonstratorConfig()
        c = env.controller
        self.origins = (env.data.site_xpos[c.grip_ids]-c.center) @ c.axis_frame
        self.rotations = env.data.site_xmat[c.grip_ids].reshape(2, 3, 3).copy()
        self.initial_angle = env.wheel_angle
        self.hand_target = c.closed_hand.copy()
        self.support = np.ones(2, dtype=bool)
        if env.free_side >= 0:
            self.support[env.free_side] = False

    def predict(self, observation, deterministic=True):
        e, cfg = self.env, self.config
        c = e.controller
        future = min(e.time+cfg.preview_seconds, (len(e.command_angles)-1)*e.dt)
        index = future/e.dt
        i = int(index)
        j = min(i+1, len(e.command_angles)-1)
        reference = e.command_angles[i]+(index-i)*(e.command_angles[j]-e.command_angles[i])
        velocity = e.command_velocities[i]+(index-i)*(e.command_velocities[j]-e.command_velocities[i])
        omega = velocity+cfg.angle_gain*(reference-e.wheel_angle)
        local = (c.wrist_positions-c.center) @ c.axis_frame
        action = np.zeros(24)
        for side in range(2):
            radius = np.linalg.norm(local[side, 1:])
            initial_radius = np.linalg.norm(self.origins[side, 1:])
            phi = np.arctan2(local[side, 2], local[side, 1])
            phi0 = np.arctan2(self.origins[side, 2], self.origins[side, 1])
            wanted_phase = phi0+(e.wheel_angle-self.initial_angle if self.support[side] else 0.)
            phase_error = np.arctan2(np.sin(wanted_phase-phi), np.cos(wanted_phase-phi))
            # The extra phase term removes wrist/command slip; no physical state
            # is assigned and all motion passes through the ordinary executor.
            phase_speed = omega if self.support[side] else cfg.position_gain*phase_error
            action[6*side:6*side+3] = [radius*phase_speed/c.config.wrist_linear_speed,
                cfg.position_gain*(initial_radius-radius)/c.config.wrist_linear_speed,
                cfg.position_gain*(self.origins[side, 0]-local[side, 0])/c.config.wrist_linear_speed]
            angle = e.wheel_angle-self.initial_angle if self.support[side] else 0.
            desired_rotation = c.axis_frame @ Rotation.from_rotvec([angle, 0., 0.]).as_matrix() @ c.axis_frame.T @ self.rotations[side]
            error = c.axis_frame.T @ Rotation.from_matrix(desired_rotation @ c.wrist_rotations[side].T).as_rotvec()
            angular = cfg.orientation_gain*error
            angular[0] += omega if self.support[side] else 0.
            action[6*side+3:6*side+6] = angular/np.deg2rad(c.config.wrist_angular_speed_deg)
        measured = e.data.qpos[c.hand_q]
        current = c.target[14:]
        # Convert an explicit preparation target to the leaky-increment action.
        action[12:] = (cfg.hand_gain*(self.hand_target-current)+(1-e.config.hand_leak_alpha)*(current-measured))/e.config.hand_delta_rad
        return np.clip(action, -1., 1.), None
