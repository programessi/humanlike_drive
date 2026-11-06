"""Contact geometry and reward sites for the fixed-grasp wheel task."""
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .cockpit import ROOT, build_scene, quat, vec

def build_dex_scene(output, robot_path=None):
    output = Path(output)
    build_scene(output, robot_path)
    tree = ET.parse(output)
    root = tree.getroot()
    root.set("model", "h1_2_dexterous_steering")
    # The old +20 degree column required the palms to approach near arm limits.
    # Tilt the separate RL fixture upward so the palm and local rim axes align
    # in a feasible initial grasp. Its pose must be calibrated before real use.
    root.find(".//body[@name='wheel_column']").set("quat", quat(Rotation.from_euler("y", -20, degrees=True)))
    contact = ET.SubElement(root, "contact")
    for side, prefix in (("left", "L"), ("right", "R")):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        hand_bodies = list(wrist.iter("body"))
        # CAD convex pieces overlap within a hand. Exclude those pairs explicitly;
        # inter-hand, elbow/hand and torso/hand collision remains enabled.
        for a, b in combinations(hand_bodies, 2):
            ET.SubElement(contact, "exclude", body1=a.get("name"), body2=b.get("name"))
        for body in hand_bodies:
            for geom in body.findall("geom"):
                if geom.get("contype") == "2":
                    geom.set("conaffinity", "15")
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            segment = "distal" if finger == "thumb" else "intermediate"
            body = root.find(f".//body[@name='{prefix}_{finger}_{segment}']")
            # Distal-link COM points are reproducible geometric proxies, not
            # calibrated tactile pads or an assertion of force closure.
            pos = np.fromstring(body.find("inertial").get("pos"), sep=" ")
            ET.SubElement(body, "site", name=f"{side}_{finger}_pad", pos=vec(pos),
                          size="0.003", rgba="0.1 0.7 1 0", group="4")
        elbow = root.find(f".//geom[@name='{side}_elbow_link_guard']")
        elbow.set("conaffinity", "15")
    torso = root.find(".//body[@name='torso_link']")
    ET.SubElement(torso, "geom", name="dex_torso_guard", type="box", pos="0 0 0.19",
                  size="0.10 0.12 0.17", contype="8", conaffinity="2", group="3",
                  rgba="0.2 0.2 0.2 0", mass="0")
    ET.indent(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output
