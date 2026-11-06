"""One manipulation trainer per workspace, released automatically on process exit."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile

from ..paths import ROOT


@contextmanager
def training_slot(workspace=ROOT):
    key = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"humanlike-drive-training-{os.getuid()}-{key}.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another manipulation trainer is running in this workspace; run training jobs sequentially") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

