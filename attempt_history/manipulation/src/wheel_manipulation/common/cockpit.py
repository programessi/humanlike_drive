"""Build a cockpit from the local Unitree model without modifying source assets."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ..paths import ROOT
DEFAULT_ROBOT = Path("/home/nubot/unitree/unitree_ros/robots/h1_2_description/h1_2.xml")
ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_NAMES = [f"{side}_{suffix}_joint" for side in ("left", "right") for suffix in ARM_SUFFIXES]
HAND_SUFFIXES = ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal")
HAND_NAMES = [f"{side}_{suffix}_joint" for side in ("L", "R") for suffix in HAND_SUFFIXES]


def vec(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def quat(rotation):
    q = rotation.as_quat()
    return vec(np.r_[q[3], q[:3]])


def build_scene(output: Path | None = None, robot_path: Path | None = None) -> Path:
    output = Path(output or ROOT / "assets/cockpit_template.xml")
    robot_path = Path(robot_path or os.environ.get("H1_MODEL_PATH", DEFAULT_ROBOT)).expanduser().resolve()
    if not robot_path.is_file():
        raise FileNotFoundError(f"H1 model missing: {robot_path}; set H1_MODEL_PATH")
    source = ET.parse(robot_path).getroot()
    root = ET.Element("mujoco", model="h1_2_driving_cockpit")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", meshdir=str(robot_path.parent / "meshes"))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", solver="Newton", iterations="50", gravity="0 0 -9.81")
    ET.SubElement(root, "size", njmax="4000", nconmax="1000")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.2", armature="0.002")
    ET.SubElement(default, "geom", friction="1.2 0.02 0.002", solref="0.012 1", solimp="0.95 0.99 0.002", condim="4")
    asset = ET.SubElement(root, "asset")
    for a in source.findall("asset"):
        for mesh in a.findall("mesh"):
            asset.append(copy.deepcopy(mesh))
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1="0.12 0.14 0.17", rgb2="0.18 0.20 0.23", width="128", height="128")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="6 6", texuniform="true")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="1 -2 3", dir="-0.3 0.3 -1", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3")
    ET.SubElement(world, "light", name="fill", pos="-1 2 2.5", dir="0.4 -0.4 -1", diffuse="0.5 0.5 0.5")
    ET.SubElement(world, "geom", name="floor", type="plane", size="4 4 0.05", material="floor_mat")
    robot = copy.deepcopy(source.find("worldbody/body[@name='pelvis']"))
    robot.set("pos", "0 0 0.83")
    robot.remove(robot.find("joint[@name='floating_base_joint']"))
    # Bake the seated leg pose and torso fixation into the kinematic tree.
    for body in robot.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name", "")
            if name == "torso_joint" or "hip_" in name or "knee" in name or "ankle" in name:
                angle = -np.pi / 2 if "hip_pitch" in name else np.pi / 2 if "knee" in name else 0.0
                old = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
                rot = Rotation.from_quat(np.r_[old[1:], old[0]])
                axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
                body.set("quat", quat(rot * Rotation.from_rotvec(axis * angle)))
                body.remove(joint)
            elif name.startswith(("L_", "R_")):
                joint.set("damping", "0.03")
                joint.set("armature", "0.0002")
    # Keep CAD meshes for appearance. Use meshes for hand contact, simple links
    # for arm collision; bit masks allow robot-wheel and robot-interior contact.
    for body in robot.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("contype") == "0":
                geom.set("rgba", "0.38 0.42 0.48 1" if geom.get("mesh") != "logo_link" else "0.9 0.9 0.9 1")
            else:
                geom.set("contype", "2")
                geom.set("conaffinity", "5")
                geom.set("group", "3")
                # Disable duplicate full-link convex collisions except palms/fingers.
                if not body.get("name", "").startswith(("L_", "R_")) and not body.get("name", "").endswith("wrist_yaw_link"):
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
        name = body.get("name", "")
        if name.endswith("wrist_yaw_link"):
            side = "left" if name.startswith("left") else "right"
            ET.SubElement(body, "site", name=f"{side}_grip", pos="0.19 0 0.008", size="0.004", rgba="0.2 0.9 0.5 0")
        if name.endswith("elbow_link"):
            ET.SubElement(body, "geom", name=name + "_guard", type="capsule", fromto="0 0 0 0.20 0 0", size="0.032", contype="2", conaffinity="5", group="3", rgba="0.2 0.2 0.2 0")
    world.append(robot)
    cockpit = ET.SubElement(world, "body", name="cockpit")

    def box(name, pos, size, color, **kw):
        return ET.SubElement(cockpit, "geom", name=name, type="box", pos=vec(pos), size=vec(size), rgba=color, contype="4", conaffinity="2", **kw)

    box("seat_base", (-0.02, 0, 0.69), (.25, .29, .07), "0.09 0.11 0.15 1")
    box("seat_back", (-.23, 0, .99), (.06, .29, .29), "0.09 0.11 0.15 1")
    box("seat_headrest", (-.25, 0, 1.39), (.055, .17, .13), "0.1 0.12 0.16 1")
    box("seat_mount", (-.05, 0, .38), (.15, .18, .24), "0.2 0.22 0.25 1")
    box("footwell", (.52, 0, .12), (.45, .49, .04), "0.15 0.17 0.21 1")
    box("dashboard", (.88, 0, 1.00), (.16, .58, .08), "0.12 0.15 0.20 1")
    box("dashboard_top", (1, 0, 1.11), (.23, .60, .025), "0.09 0.11 0.14 1")
    box("instrument_panel", (.715, 0, 1.095), (.008, .23, .075), "0.025 0.045 0.055 1")
    box("center_console", (.6, -.44, .64), (.38, .1, .12), "0.15 0.18 0.23 1")
    box("left_door", (.45, .64, .72), (.65, .035, .29), "0.14 0.17 0.22 1")
    box("right_door", (.45, -.64, .72), (.65, .035, .29), "0.14 0.17 0.22 1")
    box("left_sill", (.35, .58, .40), (.75, .05, .05), "0.24 0.27 0.32 1")
    box("right_sill", (.35, -.58, .40), (.75, .05, .05), "0.24 0.27 0.32 1")
    # Open roof and windshield preserve inspection views. Forward display can
    # receive a CARLA image via a runtime MuJoCo texture.
    ET.SubElement(asset, "texture", name="front_rgb", type="2d", builtin="flat", rgb1="0.15 0.3 0.42", width="640", height="360")
    ET.SubElement(asset, "material", name="front_mat", texture="front_rgb", texuniform="false", emission="0.6")
    display_rotation = Rotation.from_matrix(np.array([[0,0,-1],[-1,0,0],[0,1,0]],dtype=float))
    ET.SubElement(cockpit, "geom", name="front_display", type="plane", pos="1.5 0 1.55", size="0.76 0.40 0.001", quat=quat(display_rotation), material="front_mat", contype="0", conaffinity="0")
    box("display_bezel", (1.515,0,1.55), (.012,.78,.42), "0.07 0.08 0.10 1")
    for side in (-1, 1):
        ET.SubElement(cockpit, "geom", name=f"a_pillar_{side}", type="capsule", fromto=vec([1.05, side*.6, 1.05, 1.4, side*.68, 1.96]), size="0.025", rgba="0.28 0.31 0.35 1", contype="0", conaffinity="0")
    for i, y in enumerate((-.10, .10)):
        ET.SubElement(cockpit, "geom", name=f"dial_{i}", type="cylinder", pos=vec([.70, y, 1.095]), size=".05 .006", euler=vec([0,np.pi/2,0]), rgba="0.1 0.7 0.8 1", contype="0", conaffinity="0")
    for name, y in (("accelerator", -.10), ("brake", .05)):
        box(name, (.91, y, .26), (.025, .045, .08), "0.32 0.35 0.39 1", euler="0 0.3 0")
    tilt = Rotation.from_euler("y", np.deg2rad(20))
    column = ET.SubElement(world, "body", name="wheel_column", pos="0.52 0 1.09", quat=quat(tilt))
    ET.SubElement(column, "geom", name="column", type="cylinder", pos="0.1 0 0", size=".045 .12", euler=vec([0,np.pi/2,0]), rgba="0.15 0.17 0.2 1", contype="4", conaffinity="2")
    wheel = ET.SubElement(column, "body", name="steering_wheel")
    ET.SubElement(wheel, "inertial", pos="0 0 0", mass="0.95", diaginertia=".013 .007 .007")
    ET.SubElement(wheel, "joint", name="wheel_joint", type="hinge", axis="1 0 0", range=vec([-np.pi*2.5,np.pi*2.5]), damping="0.15", frictionloss="0.03", armature="0")
    radius = .14
    for i in range(40):
        a, b = i*2*np.pi/40, (i+1)*2*np.pi/40
        ET.SubElement(wheel, "geom", name=f"rim_{i}", type="capsule", fromto=vec([0,radius*np.cos(a),radius*np.sin(a),0,radius*np.cos(b),radius*np.sin(b)]), size=".014", rgba="0.045 0.05 0.055 1", contype="1", conaffinity="2", friction="1.5 .02 .002", mass="0")
    for i, a in enumerate((np.pi/2, np.pi*7/6, np.pi*11/6)):
        ET.SubElement(wheel, "geom", name=f"spoke_{i}", type="capsule", fromto=vec([0,0,0,0,.122*np.cos(a),.122*np.sin(a)]), size=".009", rgba="0.28 0.31 0.35 1", contype="1", conaffinity="2", mass="0")
    ET.SubElement(wheel, "geom", name="hub", type="cylinder", size=".037 .022", euler=vec([0,np.pi/2,0]), rgba=".16 .18 .21 1", contype="1", conaffinity="2", mass="0")
    ET.SubElement(wheel, "geom", name="center_mark", type="sphere", pos="0 0 .14", size=".016", rgba="0.9 0.62 0.1 1", contype="0", conaffinity="0", mass="0")
    ET.SubElement(column, "site", name="wheel_center", pos="0 0 0", size=".004", rgba="0 0 0 0")
    equality = ET.SubElement(root, "equality")
    for side in ("L", "R"):
        pairs = [("thumb_intermediate", "thumb_proximal_pitch", 1.6), ("thumb_distal", "thumb_proximal_pitch", 2.4)]
        pairs += [(f"{f}_intermediate", f"{f}_proximal", 1) for f in ("index", "middle", "ring", "pinky")]
        for dependent, independent, multiplier in pairs:
            ET.SubElement(equality, "joint", joint1=f"{side}_{dependent}_joint", joint2=f"{side}_{independent}_joint", polycoef=f"0 {multiplier} 0 0 0", solref=".008 1")
    actuator = ET.SubElement(root, "actuator")
    for name in ARM_NAMES + HAND_NAMES:
        limit = 40 if "shoulder" in name and "yaw" not in name else 18 if name in ARM_NAMES else .35
        ET.SubElement(actuator, "motor", name=name, joint=name, ctrllimited="true", ctrlrange=vec([-limit, limit]))
    ET.SubElement(world, "camera", name="overview", pos="1.35 -2.05 1.95", xyaxes="0.875 0.484 0 -0.174 0.315 0.933", fovy="48")
    ET.SubElement(world, "camera", name="hands", pos="0.02 -0.9 1.40", xyaxes="0.90 -0.44 0 0.18 0.36 0.91", fovy="48")
    ET.SubElement(world, "camera", name="driver", pos="0.06 0 1.43", xyaxes="0 -1 0 0 0 1", fovy="80")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient=".35 .35 .35", diffuse=".6 .6 .6", specular=".2 .2 .2")
    ET.SubElement(root, "statistic", center=".4 0 1.0", extent="2.2")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "assets/cockpit_template.xml")
    parser.add_argument("--robot", type=Path)
    args = parser.parse_args()
    print(build_scene(args.output, args.robot))
