"""Local joint indexing, nominal grasp and finite motor parameters."""
import mujoco
import numpy as np
from .cockpit import ARM_NAMES, HAND_NAMES

class RobotHardware:
    def __init__(self, model):
        self.model = model
        self.ik_data = mujoco.MjData(model)
        self.arm_jids = np.array([model.joint(n).id for n in ARM_NAMES])
        self.arm_q = model.jnt_qposadr[self.arm_jids]
        self.arm_v = model.jnt_dofadr[self.arm_jids]
        self.hand_jids = np.array([model.joint(n).id for n in HAND_NAMES])
        self.hand_q = model.jnt_qposadr[self.hand_jids]
        self.hand_v = model.jnt_dofadr[self.hand_jids]
        self.motor_q = np.r_[self.arm_q, self.hand_q]
        self.motor_v = np.r_[self.arm_v, self.hand_v]
        self.limits = model.jnt_range[self.arm_jids].copy()
        self.hand_limits = model.jnt_range[self.hand_jids].copy()
        # Thumb pitch is restricted by the coupled distal joint limits.
        self.hand_limits[[1, 7], 1] = .49
        self.center_id = model.site("wheel_center").id
        self.grip_ids = [model.site(f"{s}_grip").id for s in ("left", "right")]
        mujoco.mj_forward(model, self.ik_data)
        self.center = self.ik_data.site_xpos[self.center_id].copy()
        self.axis_frame = self.ik_data.site_xmat[self.center_id].reshape(3, 3).copy()
        self.seed = np.array([-.65, .10, .1, 1.15, 0, 0, 0, -.65, -.1, -.1, 1.15, 0, 0, 0])
        self.open_hand = np.tile([.3, .1, .15, .15, .15, .15], 2)
        self.closed_hand = np.tile([.6, .42, 1.0, 1.0, 1.0, 1.0], 2)
        self.table_angles = np.linspace(-np.deg2rad(65), np.deg2rad(65), 53)
        self.table = None
        self.max_ik_error = 0.0
        self.kp = np.r_[np.tile([140, 140, 100, 100, 35, 35, 35], 2), np.full(12, 1.5)]
        self.kd = np.r_[np.tile([12, 12, 8, 8, 3, 3, 3], 2), np.full(12, .06)]


    def torque(self, data, target, gravity_compensation=True):
        tau = self.kp * (target - data.qpos[self.motor_q]) - self.kd * data.qvel[self.motor_v]
        if gravity_compensation:
            tau += data.qfrc_bias[self.motor_v]
        return np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

