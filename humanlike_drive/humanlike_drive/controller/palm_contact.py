"""Wheel contact wrench audit, including palm friction and actual slip.

Forces returned by mj_contactForce act on geom2. No forces are applied here.
The palm label denotes the vendor hand-base mesh; local positions are retained
so a hand edge cannot silently be interpreted as the central palm surface.
"""

from __future__ import annotations
import mujoco
import numpy as np


def required_wheel_contact(model, data, wheel_dof, other_constraint):
    inverse = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 1, inverse)
    # mj_rne omits rotor armature; mj_inverse adds this term separately.
    return float(
        inverse[wheel_dof]
        + model.dof_armature[wheel_dof] * data.qacc[wheel_dof]
        - data.qfrc_applied[wheel_dof]
        - data.qfrc_passive[wheel_dof]
        - data.qfrc_actuator[wheel_dof]
        - other_constraint
    )


PARTS = ("palm", "thumb", "fingers", "other")


class ContactAudit:
    def __init__(self, model):
        self.model = model
        self.names = [model.geom(i).name or "" for i in range(model.ngeom)]
        self.wheel = np.array([n.startswith("collision_") for n in self.names])
        self.rim = np.array([n.startswith("collision_rim_") for n in self.names])
        self.robot = model.geom_contype == 2
        self.side = np.array(
            [
                (
                    0
                    if n.startswith(("L_", "left_"))
                    else 1 if n.startswith(("R_", "right_")) else -1
                )
                for n in self.names
            ]
        )
        self.part = np.array(
            [
                (
                    0
                    if n
                    in ("left_wrist_yaw_link_geom_2", "right_wrist_yaw_link_geom_2")
                    or n.startswith(("left_palm_collision_", "right_palm_collision_"))
                    else (
                        1
                        if "thumb_" in n
                        else (
                            2
                            if any(
                                f + "_" in n
                                for f in ("index", "middle", "ring", "pinky")
                            )
                            else 3
                        )
                    )
                )
                for n in self.names
            ]
        )
        self.wv = model.jnt_dofadr[model.joint("wheel_joint").id]
        self.body_vel = np.zeros((model.nbody, 6))
        zero = mujoco.MjData(model)
        mujoco.mj_kinematics(model, zero)
        self.zero_frame = zero.site("wheel_frame").xmat.reshape(3, 3).copy()

    def sample(self, data, raw=False):
        m = self.model
        center = data.site("wheel_frame").xpos
        axis = data.site("wheel_frame").xmat.reshape(3, 3)[:, 0]
        force = np.zeros((2, 4))
        torque = np.zeros((2, 4, 3))  # normal, friction, contact couple
        radial = np.zeros((2, 4))
        slip2 = np.zeros(2)
        slip_weight = np.zeros(2)
        palm_position_sum = np.zeros((2, 3))
        palm_position_weight = np.zeros(2)
        palm_below_upper_deg = np.zeros(2)
        nonrim = np.zeros(2)
        illegal = 0.0
        penetration = 0.0
        points = []
        bad = []
        for b in set(
            m.geom_bodyid[self.side >= 0].tolist() + m.geom_bodyid[self.wheel].tolist()
        ):
            mujoco.mj_objectVelocity(
                m, data, mujoco.mjtObj.mjOBJ_BODY, b, self.body_vel[b], 0
            )
        for k, c in enumerate(data.contact):
            a, b = c.geom1, c.geom2
            f = np.zeros(6)
            mujoco.mj_contactForce(m, data, k, f)
            wid = a if self.wheel[a] else b if self.wheel[b] else -1
            if wid < 0:
                moving = any(
                    self.names[g].startswith(("left_", "right_", "L_", "R_"))
                    for g in (a, b)
                )
                if (self.robot[a] and self.robot[b]) or (
                    moving and (self.robot[a] or self.robot[b])
                ):
                    illegal = max(illegal, float(f[0]))
                    if f[0] > 0.05:
                        bad.append((self.names[a], self.names[b], float(f[0])))
                continue
            other = b if wid == a else a
            s = self.side[other]
            p = self.part[other]
            if self.rim[wid]:
                penetration = max(penetration, -float(c.dist))
            if s < 0:
                continue
            if not self.rim[wid]:
                nonrim[s] += f[0]
                continue
            sign = 1.0 if wid == b else -1.0
            frame = c.frame.reshape(3, 3)
            fn = sign * frame[0] * f[0]
            ft = sign * frame[1:].T @ f[1:3]
            moment = sign * frame.T @ f[3:]
            lever = c.pos - center
            r = lever - axis * (lever @ axis)
            r /= max(np.linalg.norm(r), 1e-10)
            local = (
                c.pos - data.body(("left", "right")[s] + "_wrist_yaw_link").xpos
            ) @ data.body(("left", "right")[s] + "_wrist_yaw_link").xmat.reshape(3, 3)
            if p == 0 and not (
                0.09 <= local[0] <= 0.18
                and abs(local[2]) <= 0.035
                and local[1] * (-1 if s == 0 else 1) > 0.005
            ):
                p = 3  # wrist cuff, hand edge/back are not the palm face
            force[s, p] += f[0]
            radial[s, p] -= r @ (fn + ft)
            torque[s, p] += np.array(
                [axis @ np.cross(lever, fn), axis @ np.cross(lever, ft), axis @ moment]
            )
            local = (
                c.pos - data.body(("left", "right")[s] + "_wrist_yaw_link").xpos
            ) @ data.body(("left", "right")[s] + "_wrist_yaw_link").xmat.reshape(3, 3)
            velocities = []
            for geom in (other, wid):
                bid = m.geom_bodyid[geom]
                v = self.body_vel[bid]
                velocities.append(v[3:] + np.cross(v[:3], c.pos - data.xipos[bid]))
            relative = velocities[0] - velocities[1]
            vt = relative - c.frame[:3] * (relative @ c.frame[:3])
            if p == 0 and f[0] > 0.05:
                slip2[s] += f[0] * float(vt @ vt)
                slip_weight[s] += f[0]
                palm_position_sum[s] += f[0] * c.pos
                palm_position_weight[s] += f[0]
                palm_below_upper_deg[s] = max(
                    palm_below_upper_deg[s],
                    float(
                        np.rad2deg(
                            np.arcsin(np.clip(-r @ self.zero_frame[:, 2], 0.0, 1.0))
                        )
                    ),
                )
            if raw:
                points.append(
                    np.r_[k, a, b, s, p, c.dist, c.pos, c.frame, f, local, vt]
                )
        contact_rows = data.efc_type >= int(
            mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
        )
        efc_contact = np.where(contact_rows, data.efc_force, 0.0)
        generalized_contact = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, data, generalized_contact, efc_contact)
        return dict(
            rim_force=force,
            part_torque=torque,
            radial_force=radial,
            palm_slip=np.sqrt(slip2 / np.maximum(slip_weight, 1e-12)),
            palm_contact_position=palm_position_sum
            / np.maximum(palm_position_weight[:, None], 1e-12),
            palm_contact_weight=palm_position_weight,
            palm_below_upper_deg=palm_below_upper_deg,
            nonrim=nonrim,
            illegal=illegal,
            penetration=penetration,
            torque_residual=float(torque.sum() - generalized_contact[self.wv]),
            wheel_contact_constraint=float(generalized_contact[self.wv]),
            wheel_other_constraint=float(
                data.qfrc_constraint[self.wv] - generalized_contact[self.wv]
            ),
            points=points,
            bad=bad,
        )
