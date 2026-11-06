"""Shared audited physics; no diagnostic tracking controller is imported."""
import mujoco
import numpy as np

from .audit import Fixture, verified_scene
from .clearance import Clearance


class Simulation(Fixture):
    def __init__(self, config):
        scene, _ = verified_scene(config)
        super().__init__(scene, config)
        if not self.structural_report()["passed"]:
            raise ValueError("Scene fails structural contract")
        self.clearance = Clearance(self)

    def restore(self, state):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for key in ("qpos", "qvel", "act"):
            getattr(d, key)[:] = state[key]
        d.time = 0
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        d.qacc_warmstart[:] = state["qacc_warmstart"]

    def contacts(self):
        """Wrenches in the fixed wheel frame, referenced to the wheel center."""
        wrench, collision, _ = self.read_contacts(False)
        return wrench, collision

    def contact_details(self):
        return self.read_contacts(True)[2]

    def read_contacts(self, detailed=False):
        m, d = self.model, self.data
        wrench = np.zeros((2, 6))
        forces = np.zeros(2)
        counts = np.zeros(2,dtype=int)
        collision = 0.
        local = np.zeros(6)
        for index in range(d.ncon):
            c = d.contact[index]
            a, b = m.geom_bodyid[[c.geom1, c.geom2]]
            mujoco.mj_contactForce(m, d, index, local)
            allowed = False
            if self.wheel_body in (a, b):
                other = b if a == self.wheel_body else a
                for side, bodies in enumerate(self.hand_bodies):
                    if other in bodies:
                        # MuJoCo contact wrench is on geom2; report force on wheel.
                        sign = 1 if b == self.wheel_body else -1
                        force = sign * c.frame.reshape(3, 3).T @ local[:3]
                        torque = sign * c.frame.reshape(3, 3).T @ local[3:] + np.cross(c.pos - self.center, force)
                        wrench[side] += np.r_[self.frame.T @ force, self.frame.T @ torque]
                        if detailed:
                            forces[side] += np.linalg.norm(local[:3])
                            counts[side] += 1
                        allowed = True
            if not allowed and not any(a in bodies and b in bodies for bodies in self.hand_bodies):
                collision = max(collision, float(np.linalg.norm(local[:3])))
        detail = dict(wrench=wrench.copy(),force_norm_sum=forces,contact_count=counts,
                      collision_peak_n=collision) if detailed else None
        return wrench, collision, detail

    def joint_diagnostics(self):
        m,d = self.model,self.data
        ids = np.flatnonzero(m.jnt_limited)
        values = d.qpos[m.jnt_qposadr[ids]]
        margin = np.minimum(values-m.jnt_range[ids,0],m.jnt_range[ids,1]-values)
        controlled = set(self.jids.tolist())
        violations = [dict(name=m.joint(int(joint)).name,joint_id=int(joint),
                           active_drive=bool(joint in controlled),position_rad=float(value),
                           limits_rad=m.jnt_range[joint].tolist(),penetration_rad=float(-distance))
                      for joint,value,distance in zip(ids,values,margin) if distance < -.02]
        arm = d.qpos[self.qidx[:14]]
        bounds = m.jnt_range[self.jids[:14]]
        arm_margin = np.minimum(arm-bounds[:,0],bounds[:,1]-arm)
        return dict(joint_ids=ids,native_margins=margin,arm_margins=arm_margin,
                    violations=violations,warnings=d.warning.number.copy())

    def advance(self, target, disturbance=0.):
        # Optional declared external pulse adds to centering on the passive wheel.
        self.physics_step(target, disturbance)
        mujoco.mj_forward(self.model, self.data)

    def failure(self, collision):
        m, d = self.model, self.data
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.actuator_force)):
            return "nonfinite_physics"
        if np.any(d.warning.number):
            return "mujoco_warning"
        if abs(d.qpos[self.wheel_q]) >= np.deg2rad(440):
            return "wheel_envelope"
        q = d.qpos[self.qidx[:14]]
        bounds = m.jnt_range[self.jids[:14]]
        if np.min(np.minimum(q-bounds[:, 0], bounds[:, 1]-q)) < np.deg2rad(2):
            return "arm_margin"
        limited = np.flatnonzero(m.jnt_limited)
        q = d.qpos[m.jnt_qposadr[limited]]
        # Soft constraints allow small penetration. This is a failure, not state repair.
        if np.any(q < m.jnt_range[limited, 0] - .02) or np.any(q > m.jnt_range[limited, 1] + .02):
            return "joint_limit"
        if collision > 1.:
            return "collision"
        if np.any(d.actuator_force < m.actuator_forcerange[:, 0]-1e-6) or np.any(d.actuator_force > m.actuator_forcerange[:, 1]+1e-6):
            return "actuator_cap"
        return None
