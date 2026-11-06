"""Reparameterized tanh Gaussian with a stable, exact Jacobian correction."""
import math

import torch
from torch.nn import functional as F


class TanhNormal:
    def __init__(self, mean, log_std):
        self.mean = mean
        self.log_std = log_std.clamp(-5., 1.)
        self.base = torch.distributions.Normal(mean, self.log_std.exp())

    def sample(self):
        latent = self.base.rsample()
        return latent.tanh(), latent, self.log_prob(latent)

    def log_prob(self, latent):
        correction = 2*(math.log(2.)-latent-F.softplus(-2*latent))
        return (self.base.log_prob(latent)-correction).sum(dim=-1)

    def entropy_sample(self):
        # Differential entropy of the squashed distribution, estimated with one
        # reparameterized sample. It may be negative; this is not a discrete entropy.
        return -self.log_prob(self.base.rsample())

    def deterministic(self):
        return self.mean.tanh()
