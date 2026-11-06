"""Joint-only action executor. No wheel state or reference enters this module."""
import mujoco
import numpy as np


def effective_limits(model, joint_ids):
    """Propagate scalar linear joint equalities, including passive joint limits."""
    limits = model.jnt_range.copy()
    for _ in range(model.neq + 1):
        for eq in range(model.neq):
            if model.eq_type[eq] != int(mujoco.mjtEq.mjEQ_JOINT):
                continue
            child, parent = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
            coefficients = model.eq_data[eq, :5]
            if parent < 0 or np.any(coefficients[2:] != 0) or coefficients[1] == 0:
                raise ValueError("Only nonzero linear scalar mimic constraints are supported")
            slope = coefficients[1]
            offset = (coefficients[0] + model.qpos0[model.jnt_qposadr[child]]
                      - slope * model.qpos0[model.jnt_qposadr[parent]])
            projected = np.sort((limits[child] - offset) / slope)
            limits[parent, 0] = max(limits[parent, 0], projected[0])
            limits[parent, 1] = min(limits[parent, 1], projected[1])
            projected = np.sort(slope * limits[parent] + offset)
            limits[child, 0] = max(limits[child, 0], projected[0])
            limits[child, 1] = min(limits[child, 1], projected[1])
    result = limits[joint_ids]
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("Empty effective drive range")
    return result


class Executor:
    arm_speed = .6
    arm_lead = np.deg2rad(5.)
    DIAGNOSTIC_TYPES = ('arm_lead', 'arm_bound', 'arm_slew', 'hand_slew')

    def __init__(self, model, joint_ids, diagnostics=False):
        self.diagnostics_enabled = diagnostics
        self.limits = effective_limits(model, joint_ids)
        self.bounds = self.limits + np.array([1., -1.]) * np.r_[np.full(14, np.deg2rad(2)), np.full(12, .01)][:, None]
        self.hand_mid = self.bounds[14:].mean(axis=1)
        self.hand_half = np.diff(self.bounds[14:], axis=1).ravel() / 2
        self.hand_speed = .5 * np.diff(self.limits[14:], axis=1).ravel()
        self.target = np.zeros(26)
        self.previous = np.zeros(26)
        self.filtered = np.zeros(14)
        self._reset_diagnostics()

    def _reset_diagnostics(self):
        self.diagnostic_ticks = 0
        self.step_counts = np.zeros((4,26),dtype=np.int64)
        self.episode_counts = self.step_counts.copy()
        self.current_runs = self.step_counts.copy()
        self.longest_runs = self.step_counts.copy()
        self.last_masks = self.step_counts.astype(bool)
        self.clipped_input_axes = np.zeros(26,dtype=bool)

    def reset(self, target):
        if np.any(target < self.bounds[:, 0]) or np.any(target > self.bounds[:, 1]):
            raise ValueError("Initial target outside reserved effective limits")
        self.target = target.copy()
        self.filtered[:] = 0
        self.previous = self.hold_action()
        self._reset_diagnostics()

    def hold_action(self):
        return np.r_[np.zeros(14), (self.target[14:] - self.hand_mid) / self.hand_half].astype(np.float32)

    def begin(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (26,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 26-vector")
        self.raw = action.copy()
        self.action = np.clip(action, -1, 1)
        self.filtered = .6 * self.filtered + .4 * self.action[:14]
        self.start = self.target.copy()
        self.interventions = 0
        if self.diagnostics_enabled:
            self.step_counts[:] = 0
            self.clipped_input_axes = np.abs(self.raw-self.action) > 1e-12

    def tick(self, arm_q, dt):
        old = self.target[:14]
        lower = np.maximum.reduce([self.bounds[:14, 0], arm_q - self.arm_lead, old - self.arm_speed * dt])
        upper = np.minimum.reduce([self.bounds[:14, 1], arm_q + self.arm_lead, old + self.arm_speed * dt])
        if np.any(lower > upper + 1e-12):
            return False
        desired = old + self.filtered * self.arm_speed * dt
        applied = np.clip(desired, lower, upper)
        goal = self.hand_mid + self.action[14:] * self.hand_half
        if self.diagnostics_enabled:
            masks = np.zeros((4,26),dtype=bool)
            changed = np.abs(desired-applied)>1e-12
            masks[0,:14] = changed & (np.abs(desired-arm_q)>self.arm_lead+1e-12)
            masks[1,:14] = changed & ((desired<self.bounds[:14,0]-1e-12) | (desired>self.bounds[:14,1]+1e-12))
            masks[2,:14] = changed & (np.abs(desired-old)>self.arm_speed*dt+1e-12)
            masks[3,14:] = np.abs(goal-self.target[14:])>self.hand_speed*dt+1e-12
            self.last_masks = masks
            self.step_counts += masks
            self.episode_counts += masks
            self.current_runs = (self.current_runs+1)*masks
            self.longest_runs = np.maximum(self.longest_runs,self.current_runs)
            self.diagnostic_ticks += 1
        self.interventions += int(np.any(np.abs(desired - applied) > 1e-12))
        self.target[:14] = applied
        self.target[14:] += np.clip(goal - self.target[14:], -self.hand_speed * dt, self.hand_speed * dt)
        return True

    def diagnostics(self):
        return dict(types=self.DIAGNOSTIC_TYPES,step_counts=self.step_counts.copy(),
                    episode_counts=self.episode_counts.copy(),longest_runs=self.longest_runs.copy(),
                    input_clipped_axes=self.clipped_input_axes.copy(),episode_physics_ticks=self.diagnostic_ticks)

    def finish(self, elapsed):
        self.previous = np.r_[(self.target[:14] - self.start[:14]) / max(elapsed * self.arm_speed, 1e-12),
                              (self.target[14:] - self.hand_mid) / self.hand_half]
        # Back calculation prevents filter windup at the lead/position envelope.
        self.filtered = self.previous[:14].copy()
