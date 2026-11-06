"""Reject velocity-policy checkpoints, even if their action dimension is 18."""
import json
from pathlib import Path

from . import INTERFACE_VERSION


def require_matching_checkpoint(checkpoint, scene_sha256):
    path = Path(checkpoint).resolve()
    metadata = None
    for parent in (path.parent, path.parent.parent):
        candidate = parent / "config.json"
        if candidate.is_file():
            metadata = json.loads(candidate.read_text())
            break
    if metadata is None or metadata.get("interface_version") != INTERFACE_VERSION:
        raise ValueError("A v2 contact-residual checkpoint with matching metadata is required; v1 actions have different semantics")
    if metadata.get("scene_sha256") != scene_sha256:
        raise ValueError("Checkpoint scene does not match the evaluation/training scene")
    return metadata
