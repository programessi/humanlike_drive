"""Deployment adapter for 8 x 122 proprioceptive/reference frames."""
import numpy as np

from .common.observations import actor_frame
from . import FRAME_DIM


def contact_actor_frame(*args, reference_phase, reference_velocity, **kwargs):
    frame = np.r_[actor_frame(*args, **kwargs), reference_phase, reference_velocity]
    if frame.shape != (FRAME_DIM,) or not np.isfinite(frame).all():
        raise ValueError("Expected a finite v2 actor frame")
    return frame.astype(np.float32)


def deployment_observation(history):
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != FRAME_DIM or not np.isfinite(history).all():
        raise ValueError("Expected finite [history_length, 122] actor frames")
    return {"actor": history, "privileged": np.zeros(54, dtype=np.float32)}
