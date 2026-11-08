"""Geometry-only clearance audit respecting the scene collision filters."""

from __future__ import annotations
import mujoco
import numpy as np
from humanlike_drive.env.p0_scene import ASSEMBLY_EXCLUSIONS


class WorkspaceClearance:
    def __init__(self, m):
        self.m = m
        self.names = [m.geom(i).name or "" for i in range(m.ngeom)]
        exclusions = {
            frozenset([m.body(a).id, m.body(b).id]) for a, b, _ in ASSEMBLY_EXCLUSIONS
        }
        self.pairs = [[], []]
        for side in (0, 1):
            prefixes = ("left_", "L_") if side == 0 else ("right_", "R_")
            moving = [
                i
                for i, n in enumerate(self.names)
                if n.startswith(prefixes) and m.geom_contype[i]
            ]
            seen = set()
            for a in moving:
                for b in range(m.ngeom):
                    if (
                        a == b
                        or not m.geom_contype[b]
                        or self.names[b].startswith("collision_rim_")
                    ):
                        continue
                    if not (
                        (m.geom_contype[a] & m.geom_conaffinity[b])
                        or (m.geom_contype[b] & m.geom_conaffinity[a])
                    ):
                        continue
                    ba, bb = m.geom_bodyid[[a, b]]
                    wa, wb = m.body_weldid[[ba, bb]]
                    if (
                        wa == wb
                        or m.body_weldid[m.body_parentid[wa]] == wb
                        or m.body_weldid[m.body_parentid[wb]] == wa
                    ):
                        continue
                    if frozenset([ba, bb]) in exclusions:
                        continue
                    pair = tuple(sorted((a, b)))
                    if pair not in seen:
                        self.pairs[side].append(pair)
                        seen.add(pair)
        self.pairs = [np.asarray(p, dtype=int).reshape(-1, 2) for p in self.pairs]

    def measure(self, d, side):
        m = self.m
        nearest = []
        pairs = self.pairs[side]
        lower = (
            np.linalg.norm(d.geom_xpos[pairs[:, 0]] - d.geom_xpos[pairs[:, 1]], axis=1)
            - m.geom_rbound[pairs[:, 0]]
            - m.geom_rbound[pairs[:, 1]]
        )
        for a, b in pairs[lower <= 0.05]:
            dist = float(mujoco.mj_geomDistance(m, d, a, b, 0.05, None))
            if dist < 0.05:
                nearest.append(
                    dict(pair=[self.names[a], self.names[b]], distance_mm=dist * 1000)
                )
        nearest.sort(key=lambda x: x["distance_mm"])
        return dict(
            min_obstacle_clearance_mm=nearest[0]["distance_mm"] if nearest else 50.0,
            closest_obstacles=nearest[:3],
            distance_cap_mm=50.0,
        )
