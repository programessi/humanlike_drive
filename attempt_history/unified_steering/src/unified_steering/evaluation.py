"""Task metrics that do not depend on reward weights."""
import numpy as np


class EpisodeMetrics:
    def __init__(self):
        self.errors = []
        self.interventions = 0
        self.ticks = 0

    def add(self, error, interventions, ticks):
        self.errors.append(float(error))
        self.interventions += interventions
        self.ticks += ticks

    def summary(self):
        errors = np.asarray(self.errors)
        return dict(rmse_deg=float(np.rad2deg(np.sqrt(np.mean(errors**2)))) if len(errors) else 0.,
                    peak_error_deg=float(np.rad2deg(np.max(np.abs(errors)))) if len(errors) else 0.,
                    execution_intervention_fraction=self.interventions/max(self.ticks, 1), physics_steps=self.ticks)
