"""Independent physical checks of contact sign, torque and point velocity."""

import unittest
import mujoco
import numpy as np
from humanlike_drive.controller.palm_contact import ContactAudit, required_wheel_contact

XML = """<mujoco><option gravity="0 0 0"/><worldbody>
<body name="g29_wheel"><joint name="wheel_joint" axis="1 0 0" frictionloss=".015" armature=".002"/>
<geom name="collision_rim_test" type="box" pos=".11 .14 0" size=".02 .01 .03" mass="1" contype="1" conaffinity="2"/>
<site name="wheel_frame"/></body>
<body name="left_wrist_yaw_link" pos="0 .158 .01"><joint name="slide" type="slide" axis="0 0 1"/>
<geom name="left_wrist_yaw_link_geom_2" type="box" pos=".11 0 0" size=".02 .01 .03" mass="1" contype="2" conaffinity="1"/></body>
<body name="right_wrist_yaw_link" pos="0 -1 0"><geom size=".01" mass="1" contype="0" conaffinity="0"/></body>
</worldbody></mujoco>"""


class ContactPhysicsTest(unittest.TestCase):
    def test_world_wrench_matches_generalized_wheel_contact_force(self):
        m = mujoco.MjModel.from_xml_string(XML)
        d = mujoco.MjData(m)
        d.qvel[0] = 0.3
        d.qvel[1] = 0.1
        mujoco.mj_forward(m, d)
        a = ContactAudit(m).sample(d, True)
        self.assertGreater(a["rim_force"][0, 0], 0)
        self.assertLess(abs(a["torque_residual"]), 1e-10)
        self.assertGreater(abs(a["part_torque"][0, 0, 1]), 1e-4)

    def test_wheel_balance_includes_friction_constraint_and_rotor_armature(self):
        m = mujoco.MjModel.from_xml_string(XML)
        d = mujoco.MjData(m)
        d.qvel[:] = [0.3, 0.1]
        d.qfrc_applied[0] = 0.1
        mujoco.mj_forward(m, d)
        audit = ContactAudit(m)
        a = audit.sample(d)
        required = required_wheel_contact(m, d, audit.wv, a["wheel_other_constraint"])
        self.assertAlmostEqual(required, a["part_torque"].sum(), places=9)

    def test_slip_point_velocity_matches_independent_contact_jacobian(self):
        m = mujoco.MjModel.from_xml_string(XML)
        d = mujoco.MjData(m)
        d.qvel[:] = [0.3, 0.1]
        mujoco.mj_forward(m, d)
        a = ContactAudit(m).sample(d, True)
        for row in a["points"]:
            g1, g2 = map(int, row[1:3])
            p = row[6:9]
            normal = row[9:12]
            jac = []
            for geom in (g1, g2):
                jp = np.zeros((3, m.nv))
                jr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, p, m.geom_bodyid[geom])
                jac.append(jp @ d.qvel)
            # geom1 is wheel here; audit records hand minus wheel.
            expected = jac[1] - jac[0]
            expected -= normal * (normal @ expected)
            np.testing.assert_allclose(row[-3:], expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
