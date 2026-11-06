"""Separate fixture with force-limited PD integrated by MuJoCo implicitfast."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .common.geometry import build_dex_scene
from .paths import ROOT
from .common.cockpit import vec

SCENE_PATH = ROOT / "assets/wheel.xml"


def build_scene(output=SCENE_PATH, robot_path=None):
    output = Path(output)
    build_dex_scene(output, robot_path)
    tree = ET.parse(output)
    root = tree.getroot()
    root.set("model", "h1_2_contact_residual_implicit_pd")
    kp = np.r_[np.tile([140, 140, 100, 100, 35, 35, 35], 2), np.full(12, 1.5)]
    kd = np.r_[np.tile([12, 12, 8, 8, 3, 3, 3], 2), np.full(12, .06)]
    for i, actuator in enumerate(root.find("actuator")):
        # Same finite spring/damper and torque cap as the explicit motor law.
        # Affine velocity bias lets implicitfast include damping derivatives.
        actuator.tag = "general"
        actuator.set("ctrllimited", "false")
        actuator.set("forcelimited", "true")
        actuator.set("forcerange", actuator.get("ctrlrange"))
        actuator.set("gainprm", str(kp[i]))
        actuator.set("biastype", "affine")
        actuator.set("biasprm", vec([0, -kp[i], -kd[i]]))
    ET.indent(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    print(build_scene())
