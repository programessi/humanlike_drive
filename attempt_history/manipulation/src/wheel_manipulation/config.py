from pathlib import Path

import yaml

from .paths import ROOT
from .controller import ResidualControlConfig
from .env import ContactTaskConfig

DEFAULT_CONFIG = ROOT / "configs/task.yaml"


def load_config(path=DEFAULT_CONFIG, stage="reach10"):
    payload = yaml.safe_load(Path(path).read_text())
    config = ContactTaskConfig(**dict(payload["task"], **payload["stages"][stage]),
                               controller=ResidualControlConfig(**payload["controller"]))
    config.validate()
    return config, payload
