"""Explicit upper-rim relay schedule for arbitrary unwrapped angle waypoints.

Each hand uses its own reachable upper sector. Contact transfer is verified by
measured forces in the rollout; this schedule alone is not success evidence.
"""

from __future__ import annotations
import numpy as np
from humanlike_drive.controller.p0_control import RoundTrip, smooth_fraction


class RelayPlan:
    def __init__(self, goals_deg, stroke_deg=60.0, speed_deg_s=5.0):
        if (
            not goals_deg
            or not np.isfinite(goals_deg).all()
            or not (0 < stroke_deg <= 90)
            or speed_deg_s <= 0
        ):
            raise ValueError(
                "Finite waypoints, a 0--90 degree stroke and positive speed are required"
            )
        self.segments = []
        self.handovers = []
        self.duration = 0.0
        self.stroke = np.deg2rad(stroke_deg)
        self.ref = 0.0
        self.phases = np.zeros(2)
        self.retreat = np.zeros(2)
        first = next((g for g in goals_deg if abs(g) > 1e-8), 1.0)
        self.goal = np.deg2rad(goals_deg[0])
        self.active = 0 if first > 0 else 1
        self.support = np.zeros(2, dtype=bool)
        self.support[self.active] = True
        self._add(
            3.0,
            "initial_release",
            retreat=np.array([float(self.active != s) for s in (0, 1)]),
        )
        self._add(2.0, "hold")
        for goal_deg in goals_deg:
            goal = np.deg2rad(goal_deg)
            self.goal = goal
            while abs(goal - self.ref) > 1e-7:
                direction = np.sign(goal - self.ref)
                bounds = (0.0, self.stroke) if self.active == 0 else (-self.stroke, 0.0)
                capacity = (
                    bounds[1] - self.phases[self.active]
                    if direction > 0
                    else self.phases[self.active] - bounds[0]
                )
                if capacity < np.deg2rad(0.01):
                    old, new = self.active, 1 - self.active
                    phase_start = (
                        (0.0 if direction > 0 else self.stroke)
                        if new == 0
                        else (-self.stroke if direction > 0 else 0.0)
                    )
                    nextphases = self.phases.copy()
                    nextphases[new] = phase_start
                    self._add(4.0, "repark", phases=nextphases)
                    begin = self.duration
                    retreats = self.retreat.copy()
                    retreats[new] = 0.0
                    self._add(4.0, "approach", retreat=retreats, new=new, old=old)
                    self._add(3.0, "settle", new=new, old=old)
                    both = np.ones(2, dtype=bool)
                    self._add(3.0, "transfer", support=both, new=new, old=old)
                    transfer_end = self.duration
                    support = np.zeros(2, dtype=bool)
                    support[new] = True
                    retreats = self.retreat.copy()
                    retreats[old] = 1.0
                    self._add(
                        4.0,
                        "release",
                        retreat=retreats,
                        support=support,
                        new=new,
                        old=old,
                    )
                    solo_begin = self.duration
                    self.active = new
                    self._add(10.0, "solo", new=new, old=old)
                    self.handovers.append(
                        dict(
                            new=new,
                            old=old,
                            begin=begin,
                            transfer_end=transfer_end,
                            solo_begin=solo_begin,
                            end=self.duration,
                            reference=self.ref,
                        )
                    )
                    continue
                delta = direction * min(capacity, abs(goal - self.ref))
                nextphases = self.phases.copy()
                nextphases[self.active] += delta
                motion = RoundTrip(
                    np.rad2deg(self.ref + delta),
                    start_deg=np.rad2deg(self.ref),
                    speed_deg_s=speed_deg_s,
                    initial_hold_s=0.0,
                )
                self._add(
                    motion.move,
                    "move",
                    ref=self.ref + delta,
                    phases=nextphases,
                    motion=motion,
                )
            self._add(3.0, "goal_hold")

    def _add(
        self, duration, kind, ref=None, phases=None, retreat=None, support=None, **extra
    ):
        end_ref = self.ref if ref is None else ref
        end_phase = self.phases.copy() if phases is None else phases.copy()
        end_retreat = self.retreat.copy() if retreat is None else retreat.copy()
        end_support = self.support.copy() if support is None else support.copy()
        self.segments.append(
            dict(
                start=self.duration,
                end=self.duration + duration,
                kind=kind,
                ref0=self.ref,
                ref1=end_ref,
                phases0=self.phases.copy(),
                phases1=end_phase,
                retreat0=self.retreat.copy(),
                retreat1=end_retreat,
                support=end_support.copy(),
                active=self.active,
                goal=self.goal,
                **extra,
            )
        )
        self.duration += duration
        self.ref = end_ref
        self.phases = end_phase
        self.retreat = end_retreat
        self.support = end_support

    def extend_stationary_wait(self, entering_segment, seconds):
        """Keep a settled contact/transfer state while preserving past motion."""
        if entering_segment <= 0 or seconds <= 0:
            raise ValueError(
                "A positive wait before an existing transition is required"
            )
        previous = self.segments[entering_segment - 1]
        if previous["kind"] not in ("settle", "transfer"):
            raise ValueError("Only constant settled/transfer states may be extended")
        boundary = self.segments[entering_segment]["start"]
        previous["end"] += seconds
        for segment in self.segments[entering_segment:]:
            segment["start"] += seconds
            segment["end"] += seconds
        for event in self.handovers:
            for key in ("begin", "transfer_end", "solo_begin", "end"):
                if event[key] >= boundary - 1e-9:
                    event[key] += seconds
        self.duration += seconds

    def sample(self, t):
        i = min(
            np.searchsorted([s["end"] for s in self.segments], t, side="right"),
            len(self.segments) - 1,
        )
        s = self.segments[i]
        u = np.clip((t - s["start"]) / (s["end"] - s["start"]), 0.0, 1.0)
        alpha = (
            s["motion"]._fraction(t - s["start"])
            if s["kind"] == "move"
            else smooth_fraction(u)
        )
        ref = s["ref0"] + alpha * (s["ref1"] - s["ref0"])
        phases = s["phases0"] + alpha * (s["phases1"] - s["phases0"])
        retreat = s["retreat0"] + smooth_fraction(u) * (s["retreat1"] - s["retreat0"])
        phase = (
            1
            if s["kind"] == "move"
            else (
                7
                if s["kind"] in ("approach", "settle")
                else (
                    8
                    if s["kind"] == "transfer"
                    else (
                        9
                        if s["kind"] == "release"
                        else 10 if s["kind"] == "solo" else 2
                    )
                )
            )
        )
        return dict(
            reference=ref,
            hand_phases=phases,
            retreat=retreat,
            support=s["support"],
            phase=phase,
            segment=i,
            kind=s["kind"],
            new=s.get("new"),
            old=s.get("old"),
            active=s["active"],
            goal=s["goal"],
        )
