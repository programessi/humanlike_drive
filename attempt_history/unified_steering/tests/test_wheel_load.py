"""Physical properties of the load, including multi-turn and release behavior."""
import math
import unittest

import mujoco

from unified_steering.wheel_load import WheelLoadConfig


class WheelLoadTests(unittest.TestCase):
    def test_restores_accumulated_angle_and_saturates(self):
        load = WheelLoadConfig()
        for degrees in (-450, -360, -90, 0, 90, 360, 450):
            angle = math.radians(degrees)
            torque = load.centering_torque(angle)
            self.assertLessEqual(angle * torque, 0.)
            self.assertLessEqual(abs(torque), load.centering_torque_limit_nm)
        self.assertLess(load.centering_torque(2 * math.pi), -.9)
        self.assertGreater(abs(load.centering_torque(2 * math.pi)), abs(load.centering_torque(math.pi / 2)))

    def test_center_is_fixed_and_not_a_wrapped_target(self):
        load = WheelLoadConfig(center_angle_deg=30)
        self.assertAlmostEqual(load.centering_torque(math.radians(30)), 0.)
        self.assertGreater(load.centering_torque(0.), 0.)
        self.assertLess(load.centering_torque(math.radians(390)), 0.)

    def test_potential_matches_work(self):
        load = WheelLoadConfig()
        epsilon = 1e-5
        for angle in (-6., -1., .3, 6.):
            derivative = (load.spring_energy(angle + epsilon) - load.spring_energy(angle - epsilon)) / (2 * epsilon)
            self.assertAlmostEqual(-derivative, load.centering_torque(angle), places=7)

    def test_invalid_parameters_and_legacy_config(self):
        for options in ({"centering_stiffness_nm_per_rad": -1},
                        {"centering_torque_limit_nm": 0},
                        {"viscous_damping_nm_s_per_rad": -1},
                        {"coulomb_friction_nm": float("nan")},
                        {"center_angle_deg": float("inf")}):
            with self.assertRaises(ValueError):
                WheelLoadConfig(**options)
        self.assertEqual(WheelLoadConfig.from_config({}).centering_torque(1.), 0.)

    def test_free_wheel_returns_and_dissipates_energy(self):
        # A no-contact rotor with the actual fixture's wheel inertia.
        for sign in (-1, 1):
            model = mujoco.MjModel.from_xml_string('''<mujoco>
              <option timestep="0.002" integrator="implicitfast" gravity="0 0 0"/>
              <worldbody><body><joint name="wheel" axis="1 0 0"/>
              <inertial pos="0 0 0" mass="0.95" diaginertia=".013 .007 .007"/>
              </body></worldbody></mujoco>''')
            load = WheelLoadConfig()
            load.configure_joint(model, 0)
            data = mujoco.MjData(model)
            data.qpos[0] = sign * math.pi / 2
            mujoco.mj_forward(model, data)
            initial_energy = load.spring_energy(float(data.qpos[0]))
            for _ in range(2500):
                data.qfrc_applied[0] = load.centering_torque(float(data.qpos[0]))
                mujoco.mj_step(model, data)
            self.assertLess(abs(data.qpos[0]), math.radians(10))
            self.assertEqual(model.nu, 0)
            final_energy = load.spring_energy(float(data.qpos[0])) + .5 * .013 * data.qvel[0] ** 2
            self.assertLess(final_energy, .02 * initial_energy)


if __name__ == "__main__":
    unittest.main()
