"""Gradual dynamics randomization for the current manipulation curriculum."""
from stable_baselines3.common.callbacks import BaseCallback

class RandomizationRamp(BaseCallback):
    def __init__(self, budget, initial_steps):
        super().__init__()
        self.budget, self.initial_steps = budget, initial_steps

    def _on_step(self):
        strength = min(1., (self.num_timesteps - self.initial_steps) / max(1., self.budget * .6))
        if self.n_calls % 100 == 0:
            self.training_env.env_method("set_randomization_strength", strength)
        return True
