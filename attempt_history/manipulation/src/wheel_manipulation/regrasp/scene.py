"""Restore cross-finger collisions without changing preserved task assets."""
import xml.etree.ElementTree as ET
from pathlib import Path

from ..paths import ROOT
from ..scene import build_scene as build_preserved_scene

SCENE_PATH = ROOT / "assets/regrasp_stage1/wheel.xml"


def finger_branch(name):
    for side in ("L", "R"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            prefix = f"{side}_{finger}"
            if name.startswith(prefix + "_"):
                return prefix
    return None


def build_scene(path=SCENE_PATH):
    path = Path(path)
    build_preserved_scene(path)
    tree = ET.parse(path)
    root = tree.getroot()
    for item in list(root.find("contact")):
        if item.tag == "exclude":
            a, b = finger_branch(item.get("body1", "")), finger_branch(item.get("body2", ""))
            if a and b and a != b:
                root.find("contact").remove(item)
    root.set("model", "h1_2_bimanual_regrasp_v1")
    ET.indent(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
