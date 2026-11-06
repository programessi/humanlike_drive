"""Distances for the existing enabled collision geometry; not a full CAD safety proof."""
import mujoco
import numpy as np


class Clearance:
    def __init__(self, fixture):
        self.model = m = fixture.model
        enabled = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.hand = [np.array([g for g in range(m.ngeom)
                               if enabled[g] and m.geom_bodyid[g] in bodies], dtype=int)
                     for bodies in fixture.hand_bodies]
        self.wheel = np.flatnonzero(m.geom_bodyid == fixture.wheel_body)
        self.assembly = [np.r_[self.hand[s], m.geom(f"{side}_elbow_link_guard").id]
                         for s, side in enumerate(("left", "right"))]
        self.other = []
        for s in range(2):
            own = set(self.assembly[s].tolist())
            pairs = []
            for a in self.assembly[s]:
                for b in np.flatnonzero(enabled):
                    if b in own or b in self.wheel:
                        continue
                    if ((m.geom_contype[a] & m.geom_conaffinity[b])
                            or (m.geom_contype[b] & m.geom_conaffinity[a])):
                        pairs.append((int(a), int(b)))
            self.other.append(np.array(pairs, dtype=int))
        self.hand_wheel = [np.array([(int(a), int(b)) for a in hand for b in self.wheel])
                           for hand in self.hand]

    def minimum(self, data, pairs, maximum=.25):
        """Bounding-sphere culling followed by exact MuJoCo geom distance queries."""
        if not len(pairs):
            return maximum
        m = self.model
        a, b = pairs[:, 0], pairs[:, 1]
        lower = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1) - m.geom_rbound[a] - m.geom_rbound[b]
        planes = (m.geom_type[a] == int(mujoco.mjtGeom.mjGEOM_PLANE)) | (m.geom_type[b] == int(mujoco.mjtGeom.mjGEOM_PLANE))
        lower[planes] = -np.inf  # Infinite planes have no finite bounding sphere.
        best = maximum
        for index in np.argsort(lower):
            if lower[index] >= best:
                break
            value = mujoco.mj_geomDistance(m, data, int(a[index]), int(b[index]), best, None)
            best = min(best, float(value))
        return best

    def read(self, data, free_side):
        return {"free_hand_wheel_gap_m": self.minimum(data, self.hand_wheel[free_side]),
                "free_assembly_other_gap_m": self.minimum(data, self.other[free_side])}
