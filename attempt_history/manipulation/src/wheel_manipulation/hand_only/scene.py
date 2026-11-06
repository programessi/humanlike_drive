"""Extract actual hand subtrees; retain physical mimic constraints and wheel."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..paths import ROOT
from ..regrasp.scene import SCENE_PATH as SOURCE, build_scene as build_source

SCENE_PATH = ROOT / "assets/hand_only/wheel.xml"


def build_scene(path=SCENE_PATH):
    path = Path(path)
    if not SOURCE.exists():
        build_source()
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "inspire_dual_floating_hands_v1")
    # Compiling the source is only to read the known static fixture transform;
    # there is no arm IK or original policy, even during preparation.
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, data)
    site = source_model.site("wheel_center").id
    center = data.site_xpos[site].copy()
    frame = data.site_xmat[site].reshape(3, 3).copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, frame.ravel())
    world = root.find("worldbody")
    hands = []
    for side, sign in (("left", 1), ("right", -1)):
        body = deepcopy(root.find(f".//body[@name='{side}_wrist_yaw_link']"))
        for joint in list(body.findall("joint")):
            body.remove(joint)
        grip = body.find(f"site[@name='{side}_grip']")
        grip_pos = np.fromstring(grip.get("pos"), sep=" ")
        position = center + frame @ (np.array([-.002, sign * .163, 0.]) - grip_pos)
        body.set("pos", " ".join(map(str, position)))
        body.set("quat", " ".join(map(str, quat)))
        body.insert(0, ET.Element("freejoint", name=f"{side}_hand_free"))
        hands.append(body)
    world.remove(world.find("body[@name='pelvis']"))
    cockpit = world.find("body[@name='cockpit']")
    if cockpit is not None:
        world.remove(cockpit)
    world.extend(hands)
    present_bodies = {b.get("name") for b in root.iter("body")}
    for exclusion in list(root.find("contact")):
        if any(exclusion.get(key) not in present_bodies for key in ("body1", "body2")):
            root.find("contact").remove(exclusion)
    finger_joints = {j.get("name") for b in hands for j in b.iter("joint")}
    for actuator in list(root.find("actuator")):
        if actuator.get("joint") not in finger_joints:
            root.find("actuator").remove(actuator)
    meshes = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for asset in list(root.find("asset")):
        if asset.tag == "mesh" and asset.get("name") not in meshes:
            root.find("asset").remove(asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


if __name__ == "__main__":
    print(build_scene())
