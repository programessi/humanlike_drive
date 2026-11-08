"""P0 restart scene, derived without changing the inspection scene or CAD."""
from pathlib import Path
import xml.etree.ElementTree as ET

from humanlike_drive.env.logitech_scene import SCENE

ASSEMBLY_EXCLUSIONS = (
    ("torso_link", "left_shoulder_pitch_link", "direct shoulder joint assembly"),
    ("torso_link", "right_shoulder_pitch_link", "direct shoulder joint assembly"),
    ("left_wrist_yaw_link", "L_thumb_proximal", "two-axis thumb root assembly convex overlap"),
    ("right_wrist_yaw_link", "R_thumb_proximal", "two-axis thumb root assembly convex overlap"),
)


def build_scene(path, wheel_contacts=True):
    path = Path(path)
    tree = ET.parse(SCENE)
    root = tree.getroot()
    for asset in root.findall("./asset/*"):
        if asset.get("file"):
            asset.set("file", str((SCENE.parent / asset.get("file")).resolve()))
    for geom in root.iter("geom"):
        if geom.get("contype") == "2":
            geom.set("conaffinity", "7")
        if not wheel_contacts and geom.get("name", "").startswith("collision_"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for a, b, reason in ASSEMBLY_EXCLUSIONS:
        contact.append(ET.Comment(reason))
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
