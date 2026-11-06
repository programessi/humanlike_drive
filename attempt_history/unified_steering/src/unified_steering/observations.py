"""Deployable encoder: wrist FK uses only measured active arm q/dq."""
import mujoco
import numpy as np


class Observations:
    def __init__(self, simulation, executor):
        self.sim = simulation
        self.exe = executor
        self.sensor = mujoco.MjData(simulation.model)

    def actor(self, reference, time):
        s, e = self.sim, self.exe
        m, d = s.model, s.data
        q, dq = d.qpos[s.qidx].copy(), d.qvel[s.vidx].copy()
        half = np.diff(e.limits, axis=1).ravel() / 2
        normalized_q = (q - e.limits.mean(axis=1)) / half
        # No full simulator state copy: passive finger states/contact are absent.
        sensor = self.sensor
        sensor.qpos[:] = m.qpos0
        sensor.qpos[s.qidx[:14]] = q[:14]
        mujoco.mj_kinematics(m, sensor)
        mujoco.mj_comPos(m, sensor)
        positions, rotations, twists = [], [], []
        for site in s.grips:
            positions.extend(s.frame.T @ (sensor.site_xpos[site] - s.center) / .3)
            rotation = s.frame.T @ sensor.site_xmat[site].reshape(3, 3)
            rotations.extend(rotation[:, :2].T.ravel())
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, sensor, jp, jr, site)
            twists.extend(s.frame.T @ (jp[:, s.vidx[:14]] @ dq[:14]) / .3)
            twists.extend(s.frame.T @ (jr[:, s.vidx[:14]] @ dq[:14]))
        theta, omega = float(d.qpos[s.wheel_q]), float(d.qvel[s.wheel_v])
        result = np.r_[normalized_q[:14], dq[:14]/e.arm_speed, (e.target[:14]-q[:14])/e.arm_lead,
                       normalized_q[14:], dq[14:]/e.hand_speed, (e.target[14:]-q[14:])/(2*half[14:]),
                       positions, rotations, twists, e.previous,
                       theta/np.pi, omega/np.deg2rad(20), np.sin(theta), np.cos(theta),
                       (reference.preview(time)-theta)/np.pi, reference.sample(time)[1]/np.deg2rad(20),
                       .14/.3, .014/.05]
        if result.shape != (152,) or not np.isfinite(result).all():
            raise FloatingPointError("Invalid actor observation")
        return result.astype(np.float32)

    def build(self, reference, time):
        actor = self.actor(reference, time)
        s = self.sim
        wrench, _ = s.contacts()
        # Forces /20 N, torques /1 Nm; load /1 Nm, friction /1.
        wrench = wrench.copy()
        wrench[:, :3] /= 20
        m = s.model
        extra = np.r_[wrench.ravel(), s.wheel_load.centering_torque(float(s.data.qpos[s.wheel_q])),
                      m.geom_friction[s.clearance.wheel[0], 0], m.dof_damping[s.wheel_v],
                      m.dof_frictionloss[s.wheel_v], 1., 1., 0., 0.]
        critic = np.r_[actor, extra].astype(np.float32)
        if critic.shape != (172,) or not np.isfinite(critic).all():
            raise FloatingPointError("Invalid critic observation")
        return {"actor": actor, "critic": critic}
