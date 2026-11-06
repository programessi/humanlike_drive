"""Contact-independent passive steering load, evaluated at every physics step.

The centering term is a conservative, smoothly saturated spring. Viscous
damping and dry friction are supplied once by MuJoCo's joint mechanics.
All runtime angles are accumulated radians, never wrapped wheel orientations.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WheelLoadConfig:
    center_angle_deg: float = 0.0
    centering_stiffness_nm_per_rad: float = 0.3
    centering_torque_limit_nm: float = 1.0
    viscous_damping_nm_s_per_rad: float = 0.15
    coulomb_friction_nm: float = 0.03

    def __post_init__(self):
        values = (self.center_angle_deg, self.centering_stiffness_nm_per_rad,
                  self.centering_torque_limit_nm, self.viscous_damping_nm_s_per_rad,
                  self.coulomb_friction_nm)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Wheel load parameters must be finite")
        if min(values[1], values[3], values[4]) < 0 or values[2] <= 0:
            raise ValueError("Stiffness/damping/friction must be nonnegative; torque limit must be positive")

    @classmethod
    def from_config(cls, config):
        # Historical S00 configurations without this block reproduce spring-off runs.
        return cls(**config.get("wheel_load", {"centering_stiffness_nm_per_rad": 0.0}))

    def configure_joint(self, model, joint_id):
        if model.jnt_stiffness[joint_id] != 0:
            raise ValueError("Native wheel spring must be zero to avoid double-counting centering")
        dof = int(model.jnt_dofadr[joint_id])
        model.dof_damping[dof] = self.viscous_damping_nm_s_per_rad
        model.dof_frictionloss[dof] = self.coulomb_friction_nm

    def centering_torque(self, angle_rad):
        if not math.isfinite(angle_rad):
            raise ValueError("Wheel angle must be finite")
        error = angle_rad - math.radians(self.center_angle_deg)
        return -self.centering_torque_limit_nm * math.tanh(
            self.centering_stiffness_nm_per_rad * error / self.centering_torque_limit_nm)

    def spring_energy(self, angle_rad):
        """Stable potential energy with zero at the fixed mechanical center."""
        if not math.isfinite(angle_rad):
            raise ValueError("Wheel angle must be finite")
        stiffness = self.centering_stiffness_nm_per_rad
        if stiffness == 0:
            return 0.0
        x = stiffness * (angle_rad - math.radians(self.center_angle_deg)) / self.centering_torque_limit_nm
        log_cosh = float(np.logaddexp(x, -x) - math.log(2))
        return self.centering_torque_limit_nm ** 2 / stiffness * log_cosh
