"""Time-indexed commands, physical preparation, optional training-only state."""
from dataclasses import dataclass, field
from pathlib import Path

from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..exploration.env import CentralizedWheelEnv, ExplorationConfig
from ..common.env import DexSteeringEnv
from . import ACTOR_DIM
from .controller import RegraspController, RegraspControlConfig
from .scene import SCENE_PATH, build_scene


@dataclass
class RegraspConfig(ExplorationConfig):
    curriculum: str = "A"
    amplitude_deg: float = 10.
    episode_seconds: float = 20.
    target_speed_deg: float = 5.
    randomization_strength: float = .1
    closure_jitter_rad: float = .025
    settle_jitter_seconds: float = .08
    shaping_weight: float = .15
    max_wheel_deg: float = 110.
    prior_task_fraction: float = .25
    free_gap_m: float = .12
    controller: RegraspControlConfig = field(default_factory=RegraspControlConfig)

    def validate(self):
        if self.curriculum not in ("A", "B", "C", "D"):
            raise ValueError("Unknown regrasp curriculum")
        if not 0 <= self.amplitude_deg <= 100 or self.episode_seconds <= 0 or self.target_speed_deg <= 0:
            raise ValueError("Invalid goal envelope")
        if not 0 <= self.randomization_strength <= 1 or self.history_steps != 1:
            raise ValueError("Invalid sensing configuration")
        if not 0 <= self.prior_task_fraction <= 1 or not 0 <= self.free_gap_m <= .135:
            raise ValueError("Invalid preparation curriculum")
        if self.policy_hz <= 0 or self.target_acceleration_deg_s2 <= 0 or self.settle_seconds < 0:
            raise ValueError("Invalid timing")
        if self.target_mode not in ("hold", "step", "sine", "sequence", "mixed"):
            raise ValueError("Invalid target mode")


class RegraspEnv(CentralizedWheelEnv):
    controller_class = RegraspController
    actor_frame_dim = ACTOR_DIM

    def __init__(self, config=None, **kwargs):
        self.preparation_phase = 0.
        self.free_side = -1
        self.reset_options = {}
        self.initial_contacts = np.zeros(2, dtype=bool)
        self.contact_age = np.zeros(2)
        self.release_age = np.zeros(2)
        self.released = np.zeros(2, dtype=bool)
        self.regrasp_count = np.zeros(2, dtype=int)
        self.material_anchor = np.zeros(2)
        self.support_preserved = np.ones(2, dtype=bool)
        self.recovery_age = np.zeros(2)
        self.recovery_count = np.zeros(2, dtype=int)
        self.new_recovery = 0
        self.previous_recovery_potential = 0.
        path = Path(kwargs.setdefault("scene_path", SCENE_PATH))
        kwargs["scene_path"] = path
        if not path.exists():
            build_scene(path)
        super().__init__(config or RegraspConfig(), **kwargs)
        self.observation_space["actor"] = spaces.Box(-np.inf, np.inf, (ACTOR_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.reset_options = dict(options or {})
        self.regrasp_count[:] = 0
        self.contact_age[:] = self.release_age[:] = 0
        self.released[:] = False
        self.support_preserved[:] = True
        self.recovery_age[:] = 0
        self.recovery_count[:] = 0
        self.new_recovery = 0
        obs, info = super().reset(seed=seed, options=None)
        self.initial_contacts = self.contact_forces > .5
        self.material_anchor = self._material_phase()
        self.previous_recovery_potential = self._recovery_potential()
        info.update(self._info(self.interval_peak, self.interval_other))
        return obs, info

    def _randomize(self):
        super()._randomize()
        stage = self.config.curriculum
        if stage != "A" and not self.reset_options.get("exact_curriculum", False) and self.np_random.random() < self.config.prior_task_fraction:
            stage = str(self.np_random.choice(list("ABCD")[:list("ABCD").index(stage)]))
        self.active_curriculum = stage
        phase = self.np_random.choice([-40., -35., 35., 40.]) if stage == "C" else 0.
        self.preparation_phase = float(self.reset_options.get("phase_deg", phase))
        # C starts with the disadvantaged hand supporting and the other hand
        # open near a useful new grasp. This is a reset curriculum, no live
        # switching schedule or hand-role input is supplied to the actor.
        free = (0 if self.preparation_phase > 0 else 1) if stage == "C" else self.np_random.integers(0, 2) if stage == "B" else -1
        self.free_side = int(self.reset_options.get("free_side", free))
        if self.free_side not in (-1, 0, 1):
            raise ValueError("free_side must be -1, 0 or 1")
        gap = float(self.reset_options.get("free_gap_m", self.config.free_gap_m))
        self.controller.initial_arm = self.controller.preparation_arm(self.preparation_phase, self.free_side, gap)
        if self.free_side >= 0:
            inds = slice(6 * self.free_side, 6 * (self.free_side + 1))
            self.controller.closed_hand[inds] = self.controller.open_hand[inds]
        # Wheel state is initialized only here, before physical settling.
        self.data.qpos[self.wheel_q] = np.deg2rad(self.preparation_phase)

    def _make_commands(self):
        count = int(np.ceil((self.config.episode_seconds + .6) / self.dt)) + 2
        t = np.arange(count) * self.dt
        start = self.wheel_angle
        amplitude_deg = self.config.amplitude_deg
        if self.active_curriculum in ("A", "B") and self.config.curriculum in ("C", "D"):
            amplitude_deg = min(amplitude_deg, 10.)
        amplitude = np.deg2rad(amplitude_deg)
        default_sign = np.sign(self.preparation_phase) if self.active_curriculum == "C" else self.np_random.choice([-1, 1])
        sign = float(self.reset_options.get("direction", default_sign))
        mode = self.reset_options.get("mode", self.config.target_mode)
        if mode == "mixed":
            mode = str(self.np_random.choice(["hold", "step", "sine", "sequence"], p=[.15, .35, .2, .3]))
        if mode == "hold":
            # Absolute calibrated zero, so hold cannot hide preparation drift.
            self.episode_mode = "hold"
            calibrated_hold = np.deg2rad(self.preparation_phase)
            return np.full(count, calibrated_hold), np.zeros(count)
        if mode == "sine":
            freq = min(.06, np.deg2rad(self.config.target_speed_deg) / max(2 * np.pi * amplitude, 1e-9))
            return start + sign * amplitude * np.sin(2 * np.pi * freq * t), sign * amplitude * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        endpoints = sign * amplitude * np.array([1., 0., -1., 0.])
        if mode == "step":
            endpoints = endpoints[:1]
        angles, velocities = np.full(count, start), np.zeros(count)
        cursor, value = .5, start
        repeats = 1 if mode == "step" or self.active_curriculum != "D" else int(np.ceil(self.config.episode_seconds / 4)) + 1
        for endpoint in np.tile(endpoints, repeats):
            if cursor > t[-1]:
                break
            delta = endpoint - value
            duration = max(1., 1.875 * abs(delta) / np.deg2rad(self.config.target_speed_deg),
                           np.sqrt(5.8 * abs(delta) / np.deg2rad(self.config.target_acceleration_deg_s2)))
            u = np.clip((t - cursor) / duration, 0., 1.)
            active = t >= cursor
            angles[active] = value + delta * (10 * u[active] ** 3 - 15 * u[active] ** 4 + 6 * u[active] ** 5)
            velocities[active] = delta / duration * (30 * u[active] ** 2 - 60 * u[active] ** 3 + 30 * u[active] ** 4)
            cursor += duration + 1.
            value = endpoint
        return angles, velocities

    def _sensor_frame(self, initialize=False):
        base = super()._sensor_frame(initialize)
        c = self.controller
        pos = self.sensor_data.site_xpos[c.grip_ids]
        actual = self.sensor_data.site_xmat[c.grip_ids].reshape(2, 3, 3)
        errors = []
        for side in range(2):
            errors.extend(c.axis_frame.T @ (c.wrist_positions[side] - pos[side]) / .02)
            errors.extend(c.axis_frame.T @ Rotation.from_matrix(c.wrist_rotations[side] @ actual[side].T).as_rotvec() / .2)
        frame = np.r_[base, errors].astype(np.float32)
        if frame.shape != (ACTOR_DIM,):
            raise RuntimeError("Regrasp observation contract violated")
        return frame

    def _material_phase(self):
        c = self.controller
        local = (self.data.site_xpos[c.grip_ids] - c.center) @ c.axis_frame
        return np.arctan2(local[:, 2], local[:, 1]) - self.wheel_angle

    def _events(self):
        contact = self.contact_forces > .5
        phase = self._material_phase()
        self.new_recovery = 0
        torque = self._wheel_contact_torques()
        tracking = abs(self.target_angle - self.wheel_angle) < np.deg2rad(5)
        moving = abs(self.target_velocity) > np.deg2rad(1)
        following = self.data.qvel[self.wheel_v] * self.target_velocity > 0
        for side in range(2):
            useful = (contact[side] and self.enclosure[side] > .3 and tracking and moving and following
                      and torque[side] * self.target_velocity > .0002)
            self.recovery_age[side] = self.recovery_age[side] + self.dt if useful else 0.
            if self.recovery_age[side] > .3 and not self.initial_contacts[side] and not self.recovery_count[side]:
                self.recovery_count[side] = 1
                self.new_recovery += 1
        for side in range(2):
            peer = 1 - side
            self.contact_age[side] = self.contact_age[side] + self.dt if contact[side] else 0.
            self.release_age[side] = self.release_age[side] + self.dt if not contact[side] else 0.
            if self.release_age[side] > .2 and self.initial_contacts[side] and not self.released[side]:
                self.released[side] = True
                self.support_preserved[side] = bool(contact[peer])
            if self.released[side]:
                self.support_preserved[side] &= bool(contact[peer])
                change = np.arctan2(np.sin(phase[side] - self.material_anchor[side]), np.cos(phase[side] - self.material_anchor[side]))
                if self.contact_age[side] > .3 and abs(change) > np.deg2rad(10) and self.support_preserved[side] and abs(self.target_angle - self.wheel_angle) < np.deg2rad(5):
                    self.regrasp_count[side] += 1
                    self.material_anchor[side] = phase[side]
                    self.released[side] = False
                elif self.contact_age[side] > .3 and not self.support_preserved[side]:
                    # A failed cycle must not disable valid future cycles.
                    self.material_anchor[side] = phase[side]
                    self.released[side] = False

    def _reward(self, previous):
        error = self.target_angle - self.wheel_angle
        speed_error = self.target_velocity - self.data.qvel[self.wheel_v]
        c = self.controller
        margin = np.minimum(self.data.qpos[c.arm_q] - c.limits[:, 0], c.limits[:, 1] - self.data.qpos[c.arm_q])
        # Temporary local orientation shaping; no fixed circular position target.
        rot = c.axis_frame @ Rotation.from_rotvec([self.wheel_angle - self.initial_wheel, 0, 0]).as_matrix() @ c.axis_frame.T
        actual = self.data.site_xmat[c.grip_ids].reshape(2, 3, 3)
        orient = Rotation.from_matrix((rot @ self.geometric_rotations) @ actual.transpose(0, 2, 1)).magnitude()
        potential = self._recovery_potential()
        recovery_progress = .5 * (.995 * potential - self.previous_recovery_potential)
        self.previous_recovery_potential = potential
        self.reward_terms = {
            "angle": float(2 * np.exp(-(error / np.deg2rad(4)) ** 2) - .3 * min((error / np.deg2rad(10)) ** 2, 9)),
            "speed": float(.5 * np.exp(-(speed_error / np.deg2rad(5)) ** 2)),
            "support": float(.1 * np.max(self.enclosure) + .1 * np.any(self.contact_forces > .5)),
            "orientation_curriculum": float(self.config.shaping_weight * np.mean(np.exp(-(orient / .3) ** 2)) if self.active_curriculum == "A" else 0.),
            "recovery_progress_curriculum": float(recovery_progress),
            "recovery_once_curriculum": float(.5 * self.new_recovery if self.active_curriculum in ("B", "C") else 0.),
            "joint_margin": float(-.05 * np.mean(np.maximum(0, (.15 - margin) / .15) ** 2)),
            "motion": float(-.01 * np.mean((self.previous_action - previous) ** 2) - .002 * np.mean(self.previous_action ** 2)),
            "load": float(-.005 * max(0, self.interval_peak - 60) - .01 * (self.interval_other + self.interval_robot_peak)),
        }
        return sum(self.reward_terms.values())

    def _recovery_potential(self):
        if getattr(self, "active_curriculum", "A") not in ("B", "C") or self.free_side < 0:
            return 0.
        c = self.controller
        local = c.axis_frame.T @ (self.data.site_xpos[c.grip_ids[self.free_side]] - c.center)
        distance = np.hypot(np.linalg.norm(local[1:]) - .163, local[0] + .002)
        # Broad non-saturated progress signal for a physically clear hand.
        # No standing proximity income that disappears on successful contact.
        return float(np.clip(1 - distance / .18, 0, 1))

    def _info(self, peak, other):
        info = DexSteeringEnv._info(self, peak, other)
        torques = self._wheel_contact_torques()
        c = self.controller
        margin = np.minimum(self.data.qpos[c.arm_q] - c.limits[:, 0], c.limits[:, 1] - self.data.qpos[c.arm_q])
        info.update(preparation_phase_deg=self.preparation_phase, preparation_free_side=self.free_side,
                    active_curriculum=getattr(self, "active_curriculum", "A"),
                    left_regrasp_events=int(self.regrasp_count[0]), right_regrasp_events=int(self.regrasp_count[1]),
                    left_recovery_events=int(self.recovery_count[0]), right_recovery_events=int(self.recovery_count[1]),
                    left_wheel_contact_torque_nm=float(torques[0]), right_wheel_contact_torque_nm=float(torques[1]),
                    other_wheel_contact_torque_nm=float(torques[2]),
                    left_joint_margin_deg=float(np.rad2deg(margin[:7].min())),
                    right_joint_margin_deg=float(np.rad2deg(margin[7:].min())),
                    left_limiting_joint=self.model.joint(c.arm_jids[np.argmin(margin[:7])]).name,
                    right_limiting_joint=self.model.joint(c.arm_jids[7 + np.argmin(margin[7:])]).name,
                    collision_constraints=c.collision_constraints, collision_qp_failures=c.collision_qp_failures,
                    left_material_phase_deg=float(np.rad2deg(self._material_phase()[0])),
                    right_material_phase_deg=float(np.rad2deg(self._material_phase()[1])),
                    workspace_clips=self.controller.workspace_clips, **{f"reward_{k}": v for k, v in self.reward_terms.items()})
        return info

    def _wheel_contact_torques(self):
        """Diagnostic contact wrench on wheel, resolved about the wheel axis.

        MuJoCo's contact frame force acts on geom2; negate it for geom1.
        These simulator quantities never enter the deployable observation.
        """
        result, wrench = np.zeros(3), np.zeros(6)
        c = self.controller
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            first, second = contact.geom1 in self.wheel_geoms, contact.geom2 in self.wheel_geoms
            if first == second:
                continue
            other = contact.geom2 if first else contact.geom1
            body = self.model.geom_bodyid[other]
            side = next((s for s, bodies in enumerate(self.hand_body_groups) if body in bodies), 2)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            rotation = contact.frame.reshape(3, 3).T
            sign = -1 if first else 1
            force, moment = sign * (rotation @ wrench[:3]), sign * (rotation @ wrench[3:])
            result[side] += np.dot(np.cross(contact.pos - c.center, force) + moment, c.axis_frame[:, 0])
        return result

    def _advance_action(self, action):
        """Execute at the original rates; collision detection is symmetric."""
        c = self.controller
        self.interval_peak = self.interval_other = self.interval_robot_peak = 0.
        self.interval_command_speed = self.interval_command_acceleration = 0.
        vector = np.zeros(6)
        for _ in range(self.substeps):
            if self.physics_steps % self.robot_substeps == 0:
                before = c.velocity.copy()
                c.update(self.data, action, self.robot_dt)
                self.interval_command_speed = max(self.interval_command_speed, float(np.max(np.abs(c.velocity[:14]))))
                acceleration = float(np.max(np.abs(c.velocity[:14] - before[:14]))) / self.robot_dt
                self.interval_command_acceleration = max(self.interval_command_acceleration, acceleration)
            self.data.ctrl[:] = c.torque(self.data, c.target)
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            _, peak, other = self._contact_metrics()
            self.interval_peak = max(self.interval_peak, peak)
            self.interval_other = max(self.interval_other, other)
            for k in range(self.data.ncon):
                contact = self.data.contact[k]
                first, second = self.model.geom_contype[[contact.geom1, contact.geom2]]
                if ((first & 2 and second & (2 | 8)) or (second & 2 and first & (2 | 8))):
                    mujoco.mj_contactForce(self.model, self.data, k, vector)
                    self.interval_robot_peak = max(self.interval_robot_peak, float(np.linalg.norm(vector[:3])))

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (24,) or not np.isfinite(action).all():
            raise ValueError("Expected 24 finite commands")
        action = np.clip(action, -1, 1)
        previous = self.previous_action.copy()
        self.action_queue.append(action.copy())
        self._advance_action(list(self.action_queue)[-1 - self.action_delay])
        self.time += self.dt
        self._update_target()
        self._grasp_metrics()
        self._events()
        self.lost_contact_seconds = 0. if np.any(self.contact_forces > .5) else self.lost_contact_seconds + self.dt
        self.previous_action = action.copy()
        reward = self._reward(previous)
        frame = self._sensor_frame()
        self.actor_queue.append(frame.copy())
        observed = list(self.actor_queue)[-1 - self.observation_delay].copy()
        observed[122:128] = (self._preview() - observed[120]) / np.deg2rad(10)
        observed[128] = self.target_velocity / np.deg2rad(20)
        self.history.append(observed)
        self._latest_frame = frame
        info = self._info(self.interval_peak, self.interval_other)
        reason = ""
        if not info["finite"]:
            reason = "nonfinite"
        elif self.interval_peak > 300 or self.interval_other > 150 or self.interval_robot_peak > 150:
            reason = "simulated_load_limit"
        elif abs(self.wheel_angle) > np.deg2rad(self.config.max_wheel_deg):
            reason = "wheel_envelope"
        elif self.lost_contact_seconds > 1. and not self.config.no_contact:
            reason = "lost_both_contacts"
        elif self.controller.qp_failures > 5:
            reason = "qp_failure"
        info["termination_reason"] = reason
        if reason:
            correction = .5 * .995 * self.previous_recovery_potential
            reward -= correction
            self.reward_terms["recovery_progress_curriculum"] -= correction
            info["reward_recovery_progress_curriculum"] = self.reward_terms["recovery_progress_curriculum"]
        return self._observation(), float(reward - (10 if reason else 0)), bool(reason), bool(self.time + 1e-9 >= self.config.episode_seconds), info
