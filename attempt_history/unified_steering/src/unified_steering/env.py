"""Minimal Gymnasium environment for unified arm/hand steering (no policy)."""
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from .audit import ROOT
from .execution import Executor
from .initialization import InitialStates
from .simulation import Simulation
from .observations import Observations
from .reference import Reference
from .rewards import DEFAULT_WEIGHTS, reward_terms, combine
from .evaluation import EpisodeMetrics
from .telemetry import PHYSICS_FIELDS, physics_row
from .disturbance import WheelDisturbance


class SteeringEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        if config is None or isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config or ROOT / "configs/m1.yaml").read_text())
        self.config = config
        self.diagnostics_enabled = bool(config.get('diagnostics',False))
        self.sim = Simulation(config)
        self.executor = Executor(self.sim.model, self.sim.jids,diagnostics=self.diagnostics_enabled)
        self.encoder = Observations(self.sim, self.executor)
        self.initial_states = InitialStates(config["initial_state_directories"])
        self.action_space = spaces.Box(-1., 1., (26,), dtype=np.float32)
        self.observation_space = spaces.Dict({key: spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
                                               for key, size in (("actor", 152), ("critic", 172))})
        self.dt = self.sim.model.opt.timestep
        self.disturbance = WheelDisturbance(config.get('wheel_disturbance',{}),self.dt)
        self.period = self.dt*self.sim.substeps
        self.weights = {**DEFAULT_WEIGHTS, **config.get("reward_weights", {})}
        if set(self.weights) != set(DEFAULT_WEIGHTS) or not np.isfinite(list(self.weights.values())).all():
            raise ValueError("Invalid reward weight names/values")
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mode = options.get("support", self.config["initial_support"])
        entry, state = self.initial_states.sample(self.np_random, mode, options.get("state_index"))
        self.sim.restore(state)
        self.executor.reset(state["target"])
        theta = float(self.sim.data.qpos[self.sim.wheel_q])
        settings = self.config["reference"]
        amplitude = np.deg2rad(settings["amplitude_deg"])
        angles = options.get("angles_rad")
        if angles is None:
            angles = [theta] if settings["mode"] == "hold" else [theta, theta+amplitude, theta-amplitude, theta]
        self.set_reference(angles)
        _, collision = self.sim.contacts()
        failure = self.sim.failure(collision)
        if failure:
            raise RuntimeError(f"Invalid initial state {entry['file']}: {failure}")
        self.steps = 0
        self.done = False
        self.metrics = EpisodeMetrics()
        self.last_obs = self.encoder.build(self.reference, 0.)
        return self.last_obs, {"initial_state": entry["file"], "support": entry["support"],
                               "initial_angle_rad": theta, "observation_valid": True}

    def set_reference(self, angles_rad):
        """Upstream absolute goals in radians. Existing physical state is untouched."""
        settings = self.config["reference"]
        self.reference = Reference(angles_rad, speed=np.deg2rad(settings["speed_deg_s"]),
                                   acceleration=np.deg2rad(settings["acceleration_deg_s2"]), hold=settings["hold_seconds"])
        self.reference_origin = float(self.sim.data.time)

    def hold_action(self):
        return self.executor.hold_action()

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before step after episode end")
        s, e = self.sim, self.executor
        previous = e.previous.copy()
        e.begin(action)
        reason = s.failure(0.)
        ticks = 0
        collision_peak = 0.
        physics_trace = []
        for _ in range(s.substeps):
            if reason:
                break
            if not e.tick(s.data.qpos[s.qidx[:14]], self.dt):
                reason = "infeasible_target_envelope"
                break
            before = float(s.data.time)
            s.advance(e.target,self.disturbance.torque(before))
            ticks += 1
            _, collision, detail = s.read_contacts(self.diagnostics_enabled)
            collision_peak = max(collision_peak, collision)
            reason = s.failure(collision)
            if self.diagnostics_enabled and np.isfinite(s.data.qpos).all() and np.isfinite(s.data.qvel).all():
                physics_trace.append(physics_row(self,detail))
            if abs(s.data.time - before - self.dt) > 1e-8:
                reason = "physics_time_reset"
            if reason:
                break
        elapsed = ticks*self.dt
        e.finish(elapsed)
        self.steps += 1
        terminated = reason is not None
        truncated = not terminated and self.steps >= round(self.config["episode_seconds"] / self.period)
        time = float(s.data.time) - self.reference_origin
        angle, velocity = self.reference.sample(time)
        error = angle-float(s.data.qpos[s.wheel_q])
        speed_error = velocity-float(s.data.qvel[s.wheel_v])
        valid = True
        try:
            observation = self.encoder.build(self.reference, time)
        except (FloatingPointError, ValueError):
            observation = {key: value.copy() for key, value in self.last_obs.items()}
            terminated, truncated, valid = True, False, False
            reason = reason or "invalid_observation"
        self.last_obs = observation
        # On numerical failure avoid reading geometry or producing NaN rewards.
        finite = np.isfinite(error) and np.isfinite(speed_error)
        if finite and valid:
            limits = e.limits[:14]
            q = s.data.qpos[s.qidx[:14]]
            margin = np.minimum(q-limits[:, 0], limits[:, 1]-q)
            threshold = np.minimum(np.deg2rad(10), .1*np.diff(limits, axis=1).ravel())
            margin_cost = float(np.mean(np.clip(1-margin/threshold, 0, 1)**2))
            gap = min(s.clearance.minimum(s.data, pairs, maximum=.03) for pairs in s.clearance.other)
            clearance_cost = float(np.clip(1-gap/.03, 0, 1)**2)
            terms = reward_terms(error, speed_error, margin_cost, clearance_cost, e.previous, previous)
        else:
            terms = {key: 0. for key in DEFAULT_WEIGHTS if key != "failure"}
        reward, contributions = combine(terms, elapsed, terminated, self.weights)
        if finite:
            self.metrics.add(error, e.interventions, ticks)
        self.done = terminated or truncated
        info = dict(termination_reason=reason, observation_valid=valid, physics_steps=ticks,
                    elapsed_seconds=elapsed, reference_time_seconds=time,
                    reference_angle_rad=angle, wheel_angle_rad=float(s.data.qpos[s.wheel_q]),
                    error_rad=float(error) if finite else None, collision_peak_n=collision_peak,
                    raw_action=e.raw.copy(), executed_action=e.previous.copy(),
                    execution_interventions=e.interventions, reward_terms=terms, reward_contributions=contributions,
                    bootstrap_allowed=not terminated)
        if self.diagnostics_enabled and valid:
            info['diagnostics'] = dict(execution=e.diagnostics(),contacts=s.contact_details(),
                                       joints=s.joint_diagnostics(),physics_fields=PHYSICS_FIELDS,
                                       physics=np.asarray(physics_trace,dtype=float).reshape(-1,len(PHYSICS_FIELDS)))
        if self.done:
            info["episode_metrics"] = self.metrics.summary()
        return observation, reward, terminated, truncated, info
