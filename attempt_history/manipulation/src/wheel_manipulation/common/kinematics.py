"""Scratch-data differential IK sweep; never writes a physical wheel state."""
import mujoco
import numpy as np

def kinematic_sweep(controller_config=None):
    from ..env import ContactResidualEnv, ContactTaskConfig
    config = ContactTaskConfig(no_contact=True, settle_seconds=0)
    if controller_config is not None:
        config.controller = controller_config
    env = ContactResidualEnv(config)
    c, model = env.controller, env.model
    data = mujoco.MjData(model)
    data.qpos[c.arm_q] = c.initial_arm
    data.qpos[c.hand_q] = c.closed_hand
    mujoco.mj_forward(model, data)
    c.reset_reference(data)
    dt = env.robot_dt
    peak_speed = peak_acceleration = peak_error = 0.0
    errors = []
    # Go through both directions and the old table's problematic 22.5-25 degrees.
    for goal in [30, -30, 0]:
        goal = np.deg2rad(goal)
        for _ in range(round(8 / dt)):
            c.phase += np.clip(goal - c.phase, -np.deg2rad(10) * dt, np.deg2rad(10) * dt)
            old_velocity = c.velocity[:14].copy()
            positions, rotations = c.poses()
            c.solve_velocity(positions, rotations, data, dt)
            # This is an explicit kinematic harness, NOT a dynamic rollout.
            # Only scratch robot qpos follows the IK command; the wheel is untouched.
            data.qpos[c.arm_q] = c.target[:14]
            mujoco.mj_forward(model, data)
            error = float(np.max(np.linalg.norm(data.site_xpos[c.grip_ids] - positions, axis=1)))
            peak_error = max(peak_error, error)
            peak_speed = max(peak_speed, float(np.max(np.abs(c.velocity[:14]))))
            peak_acceleration = max(peak_acceleration, float(np.max(np.abs(c.velocity[:14] - old_velocity))) / dt)
            errors.append(error)
    result = {"type": "kinematic_scratch_data_only", "max_position_error_mm": peak_error * 1000,
              "final_position_error_mm": errors[-1] * 1000,
              "max_joint_speed_rad_s": peak_speed, "max_joint_acceleration_rad_s2": peak_acceleration,
              "qp_failures": c.qp_failures, "safety_overrides": c.safety_overrides,
              "wheel_angle_rad": float(data.qpos[env.wheel_q]),
              "passed": bool(peak_error <= .005 and peak_speed <= c.config.arm_speed + 1e-6
                             and peak_acceleration <= c.config.arm_acceleration + 1e-6 and c.qp_failures == 0)}
    env.close()
    return result
