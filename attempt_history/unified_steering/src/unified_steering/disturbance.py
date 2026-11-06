"""Bounded, time-driven evaluation pulses; no state feedback or reset assistance."""
import math


class WheelDisturbance:
    def __init__(self, config, dt):
        self.dt = dt
        self.cap = float(config.get('max_abs_torque_nm', .05))
        if not math.isfinite(self.cap) or not 0 < self.cap <= .05:
            raise ValueError('M3-A evaluation disturbance cap must be in (0, 0.05] Nm')
        self.pulses = []
        for pulse in config.get('pulses', []):
            start, duration, torque = (float(pulse[k]) for k in ('start_s','duration_s','torque_nm'))
            if not all(math.isfinite(v) for v in (start,duration,torque)) or start < 0 or duration <= 0 or abs(torque)>self.cap:
                raise ValueError('Invalid bounded disturbance pulse')
            ticks = [round(v/dt) for v in (start,duration)]
            if ticks[1]<1 or any(abs(v-t*dt)>1e-9 for v,t in zip((start,duration),ticks)):
                raise ValueError('Disturbance timing must align to physics ticks')
            self.pulses.append((ticks[0],ticks[0]+ticks[1],torque))
        self.pulses.sort()
        if any(a[1]>b[0] for a,b in zip(self.pulses,self.pulses[1:])):
            raise ValueError('Overlapping disturbance pulses are not supported')

    def torque(self, time_s):
        tick = round(time_s/self.dt)
        return next((torque for start,end,torque in self.pulses if start<=tick<end),0.)
