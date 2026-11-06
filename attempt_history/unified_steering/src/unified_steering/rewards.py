"""Starting reward weights, separately logged and independent of evaluation gates."""
import numpy as np

DEFAULT_WEIGHTS = dict(tracking=2., speed=.5, error=-.2, margin=-.2,
                       clearance=-.1, smooth=-.02, effort=-.005, failure=-20.)


def reward_terms(error, speed_error, margin_cost, clearance_cost, action, previous):
    x = abs(error) / np.deg2rad(15)
    huber = .5*x*x if x <= 1 else x-.5
    return dict(tracking=1/(1+(error/np.deg2rad(3))**2),
                speed=1/(1+(speed_error/np.deg2rad(10))**2), error=min(huber, 4.),
                margin=margin_cost, clearance=clearance_cost,
                smooth=float(np.mean(((action-previous)/2)**2)), effort=float(np.mean(action[:14]**2)))


def combine(terms, dt, failed, weights):
    contributions = {name: dt * weights[name] * value for name, value in terms.items()}
    contributions["failure"] = weights["failure"] if failed else 0.
    return float(sum(contributions.values())), contributions
