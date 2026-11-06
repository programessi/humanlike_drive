"""Training-only candidate half-turn coordination expert, never an executor.

It outputs all 24 ordinary actions. Angle-dependent schedules are demonstration
generation only; learned-policy evaluation does not instantiate this class.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .driving_roll_integral_audit import TangentRollDemonstrator


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3-2*x)


class HalfturnDemonstrator(TangentRollDemonstrator):
    release_start_deg = 45.
    release_end_deg = 75.
    park_phase_deg = 70.
    withdrawal_m = .085
    upper_closure_rad = 0.
    upper_thumb_yaw_rad = 0.
    upper_thumb_pitch_rad = 0.
    upper_radial_press_m = 0.
    upper_force_bidirectional = False
    upper_roll_scale = 1.
    upper_orientation_fraction = 1.
    upper_roll_cap_deg = 180.
    upper_axial_offset_m = 0.
    upper_radial_offset_m = 0.
    upper_late_roll_deg = 0.
    upper_late_start_deg = 60.
    return_to_initial_park = False
    park_radial_m = .03

    def pose(self, side, angle):
        c = self.env.controller
        upper = 1 if angle < 0 else 0
        scale = self.upper_roll_scale if side == upper else 1.
        orientation_angle=angle
        if side==upper:
            measured=self.env.wheel_angle-self.initial_angle
            orientation_angle-= (1-self.upper_orientation_fraction)*(self.controlled_phase-measured)
        psi = np.deg2rad(np.interp(np.rad2deg(orientation_angle), self.grid, self.roll[side]))*scale
        if side == upper:
            blend = smoothstep((abs(np.rad2deg(angle))-self.upper_late_start_deg)/(180.-self.upper_late_start_deg))
            psi += np.sign(angle)*np.deg2rad(self.upper_late_roll_deg)*blend

        if side == upper:
            psi = np.clip(psi,-np.deg2rad(self.upper_roll_cap_deg),np.deg2rad(self.upper_roll_cap_deg))
        tangent = self.tangent[side]*(1 if side == 0 else -1)
        turn = Rotation.from_rotvec(self.normal*angle).as_matrix()
        roll = Rotation.from_rotvec(tangent*psi).as_matrix()
        orient_turn=Rotation.from_rotvec(self.normal*orientation_angle).as_matrix()
        position = c.center+turn@(self.rim[side]-c.center)+orient_turn@roll@(self.p0[side]-self.rim[side])
        if side == upper:
            blend = smoothstep((abs(np.rad2deg(angle))-15.)/45.)
            radial=turn@(self.rim[side]-c.center)/self.env.rim_radius
            position+=blend*(self.normal*self.upper_axial_offset_m+radial*self.upper_radial_offset_m)
        return position, orient_turn@roll@self.rotations[side]

    def predict(self, observation, deterministic=True):
        action, _ = super().predict(observation, deterministic)
        e, c, cfg = self.env, self.env.controller, self.config
        # Reference sign resolves the role only for this labelled expert.
        angle = e.wheel_angle-self.initial_angle
        side = 0 if angle < 0 else 1
        magnitude = abs(np.rad2deg(angle))
        opening = smoothstep((magnitude-self.release_start_deg)/(self.release_end_deg-self.release_start_deg))
        parked = np.sign(angle)*min(abs(self.controlled_phase), np.deg2rad(self.park_phase_deg))
        desired, rotation = self.pose(side, parked)
        tracking_position,tracking_rotation = desired.copy(),rotation.copy()
        withdrawal = smoothstep((magnitude-self.release_end_deg-5.)/20.)
        desired = desired-self.normal*self.withdrawal_m*withdrawal
        if self.return_to_initial_park:
            local = c.axis_frame.T@(self.p0[side]-c.center)
            radial0 = c.axis_frame@np.r_[0.,local[1:]/np.linalg.norm(local[1:])]
            park = self.p0[side]-self.normal*self.withdrawal_m+radial0*self.park_radial_m
            tracking = desired+self.normal*self.withdrawal_m*withdrawal
            desired = (1-withdrawal)*tracking+withdrawal*park
            delta = Rotation.from_matrix(self.rotations[side]@rotation.T).as_rotvec()
            rotation = Rotation.from_rotvec(withdrawal*delta).as_matrix()@rotation

        eps = 1e-4
        advancing = abs(self.controlled_phase) < np.deg2rad(self.park_phase_deg)
        reference_index = min(int(e.time/e.dt), len(e.command_velocities)-1)
        omega = e.command_velocities[reference_index]+cfg.angle_gain*(e.target_angle-e.wheel_angle)
        ff = np.zeros(3); vf = np.zeros(3)
        if advancing:
            p1,r1 = self.pose(side, parked+eps)
            vf = (p1-tracking_position)/eps*omega
            ff = Rotation.from_matrix(r1@tracking_rotation.T).as_rotvec()/eps*omega
        if self.return_to_initial_park:
            vf *= 1-withdrawal
            ff *= 1-withdrawal
        linear = vf+cfg.position_gain*(desired-c.wrist_positions[side])
        local = c.axis_frame.T@(c.wrist_positions[side]-c.center)
        phi = np.arctan2(local[2],local[1])
        tangent = c.axis_frame@np.array([0.,-np.sin(phi),np.cos(phi)])
        radial = c.axis_frame@np.array([0.,np.cos(phi),np.sin(phi)])
        action[side*6:side*6+3] = np.array([linear@tangent,linear@radial,linear@self.normal])/c.config.wrist_linear_speed
        error = Rotation.from_matrix(rotation@c.wrist_rotations[side].T).as_rotvec()
        action[side*6+3:side*6+6] = (c.axis_frame.T@(ff+cfg.orientation_gain*error))/np.deg2rad(c.config.wrist_angular_speed_deg)
        target = self.hand_target.copy()
        upper = 1-side
        digits = slice(upper*6+2, upper*6+6)
        preload = smoothstep((magnitude-15.)/25.)
        target[digits] += self.upper_closure_rad*preload
        target[upper*6] += self.upper_thumb_yaw_rad*preload
        target[upper*6+1] += self.upper_thumb_pitch_rad*preload
        force_error = np.clip((20.-e.contact_forces[upper])/20.,-1. if self.upper_force_bidirectional else 0.,1.)
        action[upper*6+1] -= cfg.position_gain*self.upper_radial_press_m*force_error*preload/c.config.wrist_linear_speed

        target = np.clip(target, c.hand_limits[:,0]+.03, c.hand_limits[:,1]-.03)
        ids = slice(side*6,(side+1)*6)
        target[ids] = (1-opening)*target[ids]+opening*c.open_hand[ids]
        current,measured = c.target[14:],e.data.qpos[c.hand_q]
        action[12:] = (cfg.hand_gain*(target-current)+(1-e.config.hand_leak_alpha)*(current-measured))/e.config.hand_delta_rad
        return np.clip(action,-1.,1.),None
