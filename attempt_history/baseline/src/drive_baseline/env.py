"""Gymnasium contact-based steering task with a passive wheel."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .controller import SteeringController
from .scene import ROOT, build_scene


@dataclass
class TaskConfig:
    amplitude_deg: float = 15.0
    target_mode: str = "step"
    episode_seconds: float = 8.0
    frequency: float = .10
    randomize: bool = False
    residual_scale: float = .08
    hand_scale: float = .20
    settle_steps: int = 60
    integral_gain: float = 0.0


class SteeringEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None, scene_path=None, camera="overview", width=960, height=540):
        super().__init__()
        self.config = config or TaskConfig()
        self.scene_path = Path(scene_path or ROOT / "assets/cockpit.xml")
        if not self.scene_path.exists():
            build_scene(self.scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.controller = SteeringController(self.model)
        cache = self.scene_path.with_suffix(".ik.npz")
        if cache.is_file() and cache.stat().st_mtime >= self.scene_path.stat().st_mtime:
            with np.load(cache) as c:
                self.controller.table = c["table"].copy()
                self.controller.max_ik_error = float(c["max_error"])
        else:
            self.controller.build_table()
            np.savez(cache, table=self.controller.table, max_error=self.controller.max_ik_error)
        self.wheel_q = self.model.joint("wheel_joint").qposadr[0]
        self.wheel_v = self.model.joint("wheel_joint").dofadr[0]
        self.wheel_body = self.model.body("steering_wheel").id
        self.wheel_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.wheel_body))
        self.hand_body_groups = []
        for side in ("left", "right"):
            wrist = self.model.body(f"{side}_wrist_yaw_link").id
            bodies = {wrist}
            for i in range(wrist + 1, self.model.nbody):
                if self.model.body_parentid[i] in bodies:
                    bodies.add(i)
            self.hand_body_groups.append(bodies)
        self.render_mode = render_mode
        self.camera, self.width, self.height = camera, width, height
        self.renderer = None
        self.render_options = mujoco.MjvOption()
        self.render_options.geomgroup[3] = 0
        self.viewer = None
        self.viewer_thread = None
        self.action_space = spaces.Box(-1, 1, shape=(26,), dtype=np.float32)
        # q, dq, reference error; previous action; target/actual angles and speeds;
        # two non-cancelling hand force summaries. All normalized explicitly.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(26*3 + 26 + 6 + 2,), dtype=np.float32)
        self.dt = .02
        self.substeps = round(self.dt / self.model.opt.timestep)
        self.nominal_damping = self.model.dof_damping[self.wheel_v]
        self.nominal_friction = self.model.dof_frictionloss[self.wheel_v]
        self.nominal_geom_friction = self.model.geom_friction.copy()
        self.previous_action = np.zeros(26)
        self.ref_angle = 0.0
        self.target_angle = 0.0
        self.target_velocity = 0.0
        self.external_target = None
        self.external_target_velocity = 0.0
        self.integral_offset = 0.0
        self.time = 0.0
        self.contact_forces = np.zeros(2)

    @property
    def wheel_angle(self):
        return float(self.data.qpos[self.wheel_q])

    def _contact_metrics(self):
        force = np.zeros(2)
        peak = 0.0
        wheel_other_force = 0.0
        vector = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 not in self.wheel_geoms and c.geom2 not in self.wheel_geoms:
                continue
            other = c.geom2 if c.geom1 in self.wheel_geoms else c.geom1
            body = self.model.geom_bodyid[other]
            mujoco.mj_contactForce(self.model, self.data, i, vector)
            norm = float(np.linalg.norm(vector[:3]))
            peak = max(peak, norm)
            hand_contact = False
            for side, bodies in enumerate(self.hand_body_groups):
                if body in bodies:
                    force[side] += norm
                    hand_contact = True
            if not hand_contact:
                wheel_other_force += norm
        self.contact_forces = force
        return force, peak, wheel_other_force

    def _update_target(self):
        amp = np.deg2rad(self.config.amplitude_deg)
        if self.external_target is not None:
            target, vel = self.external_target, self.external_target_velocity
        elif self.config.target_mode == "sine":
            w = 2*np.pi*self.config.frequency
            target, vel = amp*np.sin(w*self.time + self.phase), amp*w*np.cos(w*self.time + self.phase)
        elif self.config.target_mode == "sequence":
            section = min(int(self.time / (self.config.episode_seconds/5)), 4)
            target, vel = amp * [0, 1, 0, -1, 0][section], 0.0
        else:
            target, vel = self.step_target, 0.0
        self.target_angle, self.target_velocity = float(target), float(vel)

    def set_target(self, angle, velocity=0.0):
        self.external_target = float(np.clip(angle, -.95, .95))
        self.external_target_velocity = float(velocity)
        self._update_target()
        return self._observation()

    def _motor_target(self, action):
        # Bounded feedback leads the hands along the wheel arc; the policy only
        # affects robot joint motor targets. Wheel angle is never assigned here.
        error = self.target_angle - self.wheel_angle
        proposed=self.integral_offset+self.config.integral_gain*error*self.dt
        raw=self.wheel_angle+1.2*error+np.clip(proposed,-.2,.2)
        # Avoid winding up beyond the fixed-grip workspace. No wheel torque or
        # qpos is assigned; this only offsets the robot's hand IK reference.
        if abs(raw)<=.95 or error*raw<0:
            self.integral_offset=float(np.clip(proposed,-.2,.2))
        desired = np.clip(self.wheel_angle + 1.2*error+self.integral_offset, -.95, .95)
        self.ref_angle += float(np.clip(desired - self.ref_angle, -.6*self.dt, .6*self.dt))
        arm = self.controller.reference(self.ref_angle) + self.config.residual_scale*action[:14]
        arm = np.clip(arm, self.controller.limits[:, 0], self.controller.limits[:, 1])
        hand = self.controller.closed_hand + self.config.hand_scale*action[14:]
        hand = np.clip(hand, self.controller.hand_limits[:, 0], self.controller.hand_limits[:, 1])
        return np.r_[arm, hand]

    def _observation(self):
        target = np.r_[self.controller.reference(self.ref_angle), self.controller.closed_hand]
        c = self.controller
        wheel = [self.target_angle, self.target_velocity, self.wheel_angle, self.data.qvel[self.wheel_v], self.target_angle-self.wheel_angle, self.ref_angle]
        values = np.r_[self.data.qpos[c.motor_q], self.data.qvel[c.motor_v]/5, target-self.data.qpos[c.motor_q], self.previous_action, wheel, np.minimum(self.contact_forces, 100)/100]
        return values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.time, self.ref_angle = 0.0, 0.0
        self.external_target = None
        self.external_target_velocity = 0.0
        self.integral_offset = 0.0
        self.previous_action[:] = 0
        self.phase = float(self.np_random.uniform(-.3, .3)) if self.config.randomize else 0.0
        amp = np.deg2rad(self.config.amplitude_deg)
        self.step_target = float(self.np_random.uniform(-amp, amp))
        self.model.dof_damping[self.wheel_v] = self.nominal_damping
        self.model.dof_frictionloss[self.wheel_v] = self.nominal_friction
        self.model.geom_friction[:] = self.nominal_geom_friction
        if self.config.randomize:
            self.model.dof_damping[self.wheel_v] *= self.np_random.uniform(.7, 1.3)
            self.model.dof_frictionloss[self.wheel_v] *= self.np_random.uniform(.6, 1.4)
            self.model.geom_friction[:, 0] *= self.np_random.uniform(.75, 1.25)
        c = self.controller
        self.data.qpos[c.arm_q] = c.reference(0)
        self.data.qpos[c.hand_q] = c.open_hand
        self._set_initial_mimic()
        mujoco.mj_forward(self.model, self.data)
        # Slowly close the fingers with actuators before defining episode zero.
        for i in range(self.config.settle_steps):
            fraction = min(1.0, (i+1)/max(1, self.config.settle_steps*.65))
            hand = c.open_hand + fraction*(c.closed_hand-c.open_hand)
            motor_target = np.r_[c.reference(0), hand]
            self._advance(motor_target)
        self.ref_angle = self.wheel_angle
        self.data.time = 0.0
        self._update_target()
        forces, peak, other = self._contact_metrics()
        info = self._info(peak, other)
        return self._observation(), info

    def _set_initial_mimic(self):
        # This is reset initialization only. Runtime coupling uses equality constraints.
        for side in ("L", "R"):
            for finger in ("index", "middle", "ring", "pinky"):
                self.data.qpos[self.model.joint(f"{side}_{finger}_intermediate_joint").qposadr] = self.data.qpos[self.model.joint(f"{side}_{finger}_proximal_joint").qposadr]
            pitch = self.data.qpos[self.model.joint(f"{side}_thumb_proximal_pitch_joint").qposadr][0]
            for segment, ratio in (("intermediate", 1.6), ("distal", 2.4)):
                self.data.qpos[self.model.joint(f"{side}_thumb_{segment}_joint").qposadr] = ratio*pitch

    def _advance(self, target):
        for _ in range(self.substeps):
            self.data.ctrl[:] = self.controller.torque(self.data, target)
            mujoco.mj_step(self.model, self.data)

    def _info(self, peak, other):
        error = self.target_angle-self.wheel_angle
        return {"time": self.time, "target_angle": self.target_angle, "wheel_angle": self.wheel_angle,
                "error_deg": float(np.rad2deg(error)), "wheel_velocity": float(self.data.qvel[self.wheel_v]),
                "left_force": float(self.contact_forces[0]), "right_force": float(self.contact_forces[1]),
                "peak_force": peak, "non_hand_wheel_force": other,
                "saturation": float(np.mean(np.abs(self.data.ctrl) >= .98*self.model.actuator_ctrlrange[:, 1])),
                "success": bool(abs(error) < np.deg2rad(3) and abs(self.data.qvel[self.wheel_v]) < .1),
                "finite": bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        if action.shape != (26,) or not np.isfinite(action).all():
            raise ValueError("action must have 26 finite values")
        target = self._motor_target(action)
        self._advance(target)
        self.time += self.dt
        self._update_target()
        forces, peak, other = self._contact_metrics()
        info = self._info(peak, other)
        error = self.target_angle-self.wheel_angle
        verror = self.target_velocity-self.data.qvel[self.wheel_v]
        reward = 3*np.exp(-((error/.15)**2)) + .3*np.exp(-verror**2)
        reward += .1*np.minimum(forces/5, 1).sum()
        reward -= .02*np.square(action).mean() + .05*np.square(action-self.previous_action).mean()
        reward -= .002*max(0.0, peak-80) + .005*other
        self.previous_action = action.copy()
        terminated = not info["finite"] or peak > 300 or other > 150 or abs(self.wheel_angle) > 1.3
        truncated = self.time >= self.config.episode_seconds
        if terminated:
            reward -= 5
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                import threading
                existing_threads=set(threading.enumerate())
                self.viewer = viewer.launch_passive(self.model, self.data)
                self.viewer_thread=next((thread for thread in threading.enumerate()
                                         if thread not in existing_threads and "_launch_internal" in thread.name),None)
                self.viewer.opt.geomgroup[3] = 0
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self.model.camera(self.camera).id
            self.viewer.sync()
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.render_options)
        return self.renderer.render().copy()

    def update_display(self, rgb):
        """Show the actual CARLA RGB on the front display when rendering."""
        from PIL import Image
        if self.render_mode == "human" and self.viewer is None:
            self.render()
        elif self.render_mode != "human" and self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        tex = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "front_rgb")
        width, height = int(self.model.tex_width[tex]), int(self.model.tex_height[tex])
        pixels = np.asarray(Image.fromarray(rgb).resize((width,height)),dtype=np.uint8)
        address = self.model.tex_adr[tex]
        self.model.tex_data[address:address+width*height*3] = pixels.ravel()
        if self.render_mode == "human":
            self.viewer.update_texture(tex)
        else:
            mujoco.mjr_uploadTexture(self.model,self.renderer._mjr_context,tex)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            # MuJoCo 3.3's passive viewer closes asynchronously. Let its native
            # GL thread finish before Python tears down model/driver objects.
            if self.viewer_thread is not None:
                self.viewer_thread.join(timeout=5)
                self.viewer_thread = None
            self.viewer = None
