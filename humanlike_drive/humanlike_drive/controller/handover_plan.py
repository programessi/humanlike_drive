"""Reviewable P0 engineering schedule; no state reset or wheel actuation.

The receiving hand returns to a reachable upper-rim pose rather than copying
the departing hand. The contact-driven rollout determines whether it works.
"""
from __future__ import annotations

import numpy as np

from humanlike_drive.controller.p0_control import RoundTrip, release_motion, smooth_fraction


class HandoverPlan:
    def __init__(self, target_deg=30., moving=False):
        self.sign = 1 if target_deg > 0 else -1
        self.original = 0 if target_deg > 0 else 1
        self.receiving = 1-self.original
        self.moving = moving
        self.outbound = RoundTrip(target_deg, initial_hold_s=5.)
        self.arrival = 5.+self.outbound.move
        self.approach = self.arrival+10.
        self.transfer = self.approach+5.
        self.release = self.transfer+3.
        self.single = self.release+3.
        self.target_hold = self.single+10.
        self.return_begin = self.target_hold+3.
        self.target = np.deg2rad(target_deg)
        # 21 s of positive speed, .5 s cosine ramps; 20.5 degrees travel.
        self.drift = RoundTrip(self.sign*20.5, speed_deg_s=1., initial_hold_s=0.) if moving else None
        self.final_target = self.target+(np.deg2rad(self.sign*20.5) if moving else 0.)
        self.return_path = RoundTrip(0., start_deg=np.rad2deg(self.final_target), initial_hold_s=0.)
        self.return_end = self.return_begin+self.return_path.move
        self.duration = self.return_end+8.
        self.new_epoch = self.reference(self.approach+2.8)
        self.old_park_phase = self.reference(self.single)

    def reference(self, t):
        if t < self.arrival:
            return self.outbound.sample(t)[0]
        if t < self.return_begin:
            increment = 0.
            if self.moving:
                u = np.clip(t-self.approach, 0., self.drift.move)
                increment = self.drift.target*self.drift._fraction(u)
            return self.target+increment
        if t < self.return_end:
            f = self.return_path._fraction(t-self.return_begin)
            return self.final_target*(1.-f)
        return 0.

    def sample(self, t, closed):
        ref = self.reference(t)
        phases = np.array([ref, ref])
        retreat = np.zeros(2)
        hands = closed.copy()
        support = [True, True]
        old, new = self.original, self.receiving
        phase = 0
        if t < 2.:
            pass
        elif t < 5.:
            phase = 5
            retreat[new], hands[new] = release_motion(t-2., closed[new])
            phases[new] = 0.
        elif t < self.approach:
            phase = 1 if t < self.arrival else 6
            retreat[new], hands[new] = release_motion(3., closed[new])
            phases[new] = 0.
            support[new] = False
        elif t < self.transfer:
            phase = 7
            u = t-self.approach
            retreat[new], hands[new] = release_motion(u, closed[new], regrasp=True)
            phases[new] = max(0., self.sign*(ref-self.new_epoch))*self.sign if u >= 2.8 else 0.
            support[new] = False
        elif t < self.release:
            phase = 8
            phases[new] = ref-self.new_epoch
        elif t < self.single:
            phase = 9
            phases[new] = ref-self.new_epoch
            retreat[old], hands[old] = release_motion(t-self.release, closed[old])
        elif t < self.return_end:
            phase = 10 if t < self.target_hold else 2 if t < self.return_begin else 3
            phases[new] = ref-self.new_epoch
            retreat[old], hands[old] = release_motion(3., closed[old])
            phases[old] = (1.-smooth_fraction((t-self.single)/2.))*self.old_park_phase
            support[old] = False
        elif t < self.return_end+5.:
            phase = 11
            phases[new] = -self.new_epoch
            phases[old] = 0.
            retreat[old], hands[old] = release_motion(t-self.return_end, closed[old], regrasp=True)
            support[old] = False
        else:
            phase = 4
            phases[new] = -self.new_epoch
            phases[old] = 0.
        segment = self.target if t < self.approach else self.final_target if t < self.return_begin else 0.
        return dict(reference=ref, phase=phase, target=segment, support=support,
                    hand_phases=phases, retreat=retreat, fingers=hands)
