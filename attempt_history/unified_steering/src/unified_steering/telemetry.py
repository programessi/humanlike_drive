"""Optional read-only 500 Hz physical telemetry; never part of actor input."""
import numpy as np

PHYSICS_FIELDS = ('time_s','wheel_angle_rad','wheel_speed_rad_s','applied_centering_nm',
                  'left_contact_force_n','right_contact_force_n','left_contact_count','right_contact_count',
                  'left_contact_torque_nm','right_contact_torque_nm','collision_n','min_arm_margin_rad',
                  'max_force_cap_ratio','max_arm_target_error_rad','min_native_joint_margin_rad',
                  'arm_lead_axes','arm_bound_axes','arm_slew_axes','hand_slew_axes',
                  'reference_rad','reference_speed_rad_s','disturbance_torque_nm','total_wheel_torque_nm')


def physics_row(env, contacts):
    s,e = env.sim,env.executor
    m,d = s.model,s.data
    joints = s.joint_diagnostics()
    angle,velocity = env.reference.sample(float(d.time)-env.reference_origin)
    load = s.last_wheel_load
    return np.r_[d.time,d.qpos[s.wheel_q],d.qvel[s.wheel_v],load['centering_torque_applied_nm'],
                 contacts['force_norm_sum'],contacts['contact_count'],contacts['wrench'][:,3],
                 contacts['collision_peak_n'],np.min(joints['arm_margins']),
                 np.max(np.abs(d.actuator_force)/m.actuator_forcerange[:,1]),
                 np.max(np.abs(e.target[:14]-d.qpos[s.qidx[:14]])),np.min(joints['native_margins']),
                 e.last_masks.sum(axis=1),angle,velocity,
                 load['disturbance_torque_applied_nm'],load['total_external_wheel_torque_applied_nm']]
