"""Analytic time-driven quintic references in accumulated radians."""
import numpy as np


class Reference:
    def __init__(self, angles, speed=np.deg2rad(10), acceleration=np.deg2rad(30), hold=1., envelope=np.deg2rad(400)):
        angles = np.asarray(angles, dtype=float)
        if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
            raise ValueError("Finite nonempty angle sequence required")
        if np.max(np.abs(angles)) > envelope or min(speed, acceleration) <= 0 or hold < 0:
            raise ValueError("Reference outside envelope or invalid timing")
        self.segments = []
        time = 0.
        self.initial, self.final = angles[0], angles[-1]
        for i, angle in enumerate(angles):
            self.segments.append((time, time + hold, angle, angle))
            time += hold
            if i + 1 < len(angles):
                delta = abs(angles[i + 1] - angle)
                duration = max(1.875 * delta / speed, np.sqrt((10 / np.sqrt(3)) * delta / acceleration), 1e-6)
                self.segments.append((time, time + duration, angle, angles[i + 1]))
                time += duration
        self.duration = time

    def sample(self, time):
        if time < 0:
            return float(self.initial), 0.
        for start, end, a, b in self.segments:
            if time < end:
                u = np.clip((time - start) / max(end - start, 1e-12), 0, 1)
                blend = 10*u**3 - 15*u**4 + 6*u**5
                rate = 30*u**2*(1-u)**2 / max(end - start, 1e-12)
                return float(a + (b-a)*blend), float((b-a)*rate)
        return float(self.final), 0.

    def preview(self, time):
        return np.array([self.sample(time + offset)[0] for offset in np.arange(11) / 10])
