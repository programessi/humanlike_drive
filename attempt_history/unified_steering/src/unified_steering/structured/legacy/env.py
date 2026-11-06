"""Known initial grasp and internal wheel goals; no navigation or CARLA."""
from dataclasses import dataclass, field
from pathlib import Path

from gymnasium import spaces
import numpy as np

from .common.env import DexSteeringEnv, DexTaskConfig
from . import FRAME_DIM
from .controller import ContactResidualController, ResidualControlConfig
from .scene import SCENE_PATH, build_scene


@dataclass
class ContactTaskConfig(DexTaskConfig):
    episode_seconds: float = 30.0
    controller: ResidualControlConfig = field(default_factory=ResidualControlConfig)


class ContactResidualEnv(DexSteeringEnv):
    controller_class = ContactResidualController
    actor_frame_dim = FRAME_DIM

    def __init__(self, config=None, **kwargs):
        kwargs.setdefault("scene_path", SCENE_PATH)
        kwargs["scene_path"] = Path(kwargs["scene_path"])
        if not kwargs["scene_path"].exists():
            build_scene(kwargs["scene_path"])
        super().__init__(config or ContactTaskConfig(), **kwargs)

    def _randomize(self):
        super()._randomize()
        c, m = self.controller, self.model
        m.actuator_gainprm[:, 0] = c.kp
        m.actuator_biasprm[:, 1] = -c.kp
        m.actuator_biasprm[:, 2] = -c.kd
        # Shared randomization uses the preserved numeric motor torque ranges;
        # these become FORCE limits, never position-command clipping limits.
        m.actuator_forcerange[:] = m.actuator_ctrlrange

    def _sensor_frame(self, initialize=False):
        frame = super()._sensor_frame(initialize)
        return np.r_[frame, self.controller.nominal_phase,
                     self.controller.nominal_velocity].astype(np.float32)

    def _info(self, peak, other):
        info = super()._info(peak, other)
        c = self.controller
        info.update(common_reference_phase_deg=float(np.rad2deg(c.nominal_phase)),
                    common_reference_velocity_deg_s=float(np.rad2deg(c.nominal_velocity)),
                    common_residual_deg=float(np.rad2deg(c.phase_residual[0])),
                    differential_residual_deg=float(np.rad2deg(c.phase_residual[1])),
                    left_radial_residual_mm=float(c.radial[0] * 1000),
                    left_axial_residual_mm=float(c.axial[0] * 1000),
                    right_radial_residual_mm=float(c.radial[1] * 1000),
                    right_axial_residual_mm=float(c.axial[1] * 1000),
                    left_wrist_reference_velocity_deg_s=float(np.rad2deg(c.phase_reference_velocity[0])),
                    right_wrist_reference_velocity_deg_s=float(np.rad2deg(c.phase_reference_velocity[1])),
                    encoder_velocity_deg_s=float(np.rad2deg(c.encoder_velocity)),
                    reference_clips=c.reference_clips)
        return info

    def _advance_action(self, action):
        self.controller.set_goal(self.target_angle, self.target_velocity)
        super()._advance_action(action)

    def step(self, action):
        # Base environment handles physics, delayed actions, sensing and strict
        # grasp/hold termination. Replace only its task-specific reward.
        previous = self.previous_action.copy()
        obs, _, terminated, truncated, info = super().step(action)
        # Reference state is local software state, not a delayed physical sensor.
        self.history[-1][-2:] = [self.controller.nominal_phase, self.controller.nominal_velocity]
        obs = self._observation()
        error = self.target_angle - self.wheel_angle
        speed_error = self.target_velocity - self.data.qvel[self.wheel_v]
        contacts = self.contact_forces > .2
        reward = 3 * np.exp(-(error / np.deg2rad(2)) ** 2)
        reward += np.exp(-(error / np.deg2rad(8)) ** 2)
        reward += .5 * np.exp(-(speed_error / np.deg2rad(5)) ** 2)
        reward += .25 * float(self.enclosure.mean()) + .1 * float(contacts.mean())
        reward -= .4 * float(np.mean(np.minimum((self.slip / np.deg2rad(5)) ** 2, 4)))
        reward -= .005 * max(0., self.interval_peak - 60)
        reward -= .01 * (self.interval_other + self.interval_robot_peak)
        reward -= .03 * float(np.mean(np.square(self.previous_action - previous)))
        reward -= .02 * float(np.mean(np.square(self.previous_action)))
        if terminated:
            reward -= 10
        return obs, float(reward), terminated, truncated, info


class HandsOnlyContactEnv(ContactResidualEnv):
    """Separately train 12 finger residuals with the identical common controller."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(-1, 1, (12,), dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Hand-only v2 action must have 12 finite values")
        return super().step(np.r_[action, np.zeros(6)])
