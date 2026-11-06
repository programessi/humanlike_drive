"""Build a fresh H1-2 / Logitech G29 bench from vendor CAD, without old scenes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from humanlike_drive.env.inspire_hand import mimic_relations

PROJECT = Path(__file__).resolve().parents[2]
SCENE = PROJECT / "assets/scenes/logitech_g29.xml"
ROBOT_SOURCE = Path("/home/nubot/unitree/unitree_ros/robots/h1_2_description/h1_2.xml")
G29_SOURCE = Path("/home/nubot/g29_wheel_urdf")
ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_NAMES = [f"{side}_{suffix}_joint" for side in ("left", "right") for suffix in ARM_SUFFIXES]
HAND_SUFFIXES = ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
                 "middle_proximal", "ring_proximal", "pinky_proximal")
HAND_NAMES = [f"{side}_{suffix}_joint" for side in ("L", "R") for suffix in HAND_SUFFIXES]


def vec(x):
    return " ".join(f"{float(v):.10g}" for v in np.asarray(x).ravel())


def quat(r):
    q = r.as_quat()
    return vec(q[[3, 0, 1, 2]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera(world, name, eye, target, fovy=45):
    eye, target = np.array(eye), np.array(target)
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    ET.SubElement(world, "camera", name=name, pos=vec(eye),
                  xyaxes=vec(np.r_[right, up]), fovy=str(fovy))


def convex_piece(points, output):
    points = np.unique(np.round(points, 9), axis=0)
    if len(points) < 4 or np.linalg.svd(points - points.mean(0), compute_uv=False)[-1] < 1e-6:
        return False
    hull = ConvexHull(points)
    mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices, process=True)
    trimesh.repair.fix_normals(mesh)
    mesh.export(output)
    return True


def wheel_collision(mesh, axis, output):
    """Local convex sectors follow the sculpted rim; never hull the whole ring.

    Sector boundaries overlap by one source triangle. Inner radial layers keep
    spokes/paddles separate from the annulus and preserve the finger openings.
    """
    aligned = Rotation.align_vectors([[0, 0, 1]], [axis])[0].apply(mesh.vertices)
    radius = np.linalg.norm(aligned[:, :2], axis=1)
    angles = np.arctan2(aligned[:, 1], aligned[:, 0])
    face_angles = np.arctan2(np.sin(angles[mesh.faces]).mean(1),
                             np.cos(angles[mesh.faces]).mean(1))
    face_r = radius[mesh.faces].mean(1)
    face_z = aligned[mesh.faces, 2].mean(1)
    parts = []
    for label, lo, hi, count in [("rim", .101, .3, 96), ("inner", .045, .101, 48),
                                 ("hub", 0, .045, 12)]:
        width = 2 * np.pi / count
        # The inside of the sculpted lower grip is concave in axial section.
        # A single sector hull bridged that recess by over 2 mm. Axial layers
        # preserve it instead of filling it with a false contact surface.
        depths = ([(-.04, -.012), (-.012, -.004), (-.004, .002), (.002, .005), (.005, .015)]
                  if label == "rim" else [(-.04, -.012), (-.012, .015), (.015, .045), (.045, .12)])
        for layer, (zlo, zhi) in enumerate(depths):
            for i in range(count):
                center = -np.pi + (i + .5) * width
                delta = np.arctan2(np.sin(face_angles - center), np.cos(face_angles - center))
                selected = ((np.abs(delta) <= width * .53) & (face_r >= lo) & (face_r < hi)
                            & (face_z >= zlo) & (face_z < zhi))
                ids = np.unique(mesh.faces[selected])
                # The CAD's metal paddle surface is at axial 29.33--33.8 mm,
                # radius up to 100.44 mm, wholly separate from the rim. Give
                # its outer sectors an explicit name for contact inspection.
                part_label = "paddle" if label == "inner" and layer == 2 else label
                path = output / f"{part_label}_{layer}_{i:03d}.stl"
                if convex_piece(mesh.vertices[ids], path):
                    parts.append((path, part_label))
    return parts


def build(robot_source=ROBOT_SOURCE, g29_source=G29_SOURCE, table_height=.94,
          base_x=.64):
    robot_source, g29_source = Path(robot_source), Path(g29_source)
    robot_dir = PROJECT / "assets/robots/h1_2"
    object_dir = PROJECT / "assets/objects/logitech_g29"
    collision_dir = object_dir / "collision"
    for d in (robot_dir / "meshes", object_dir / "meshes", collision_dir, SCENE.parent):
        d.mkdir(parents=True, exist_ok=True)
    source = ET.parse(robot_source).getroot()
    shutil.copy2(robot_source, robot_dir / "source.xml")
    hand_urdf = robot_source.with_suffix(".urdf")
    shutil.copy2(hand_urdf, robot_dir / "source.urdf")
    relations = mimic_relations(hand_urdf)
    for assets in source.findall("asset"):
        for mesh in assets.findall("mesh"):
            shutil.copy2(robot_source.parent / "meshes" / mesh.get("file"),
                         robot_dir / "meshes" / mesh.get("file"))
    urdf_path = g29_source / "urdf/g29_wheel_urdf.urdf"
    shutil.copy2(urdf_path, object_dir / "source.urdf")
    for name in ("base_link_wheel.STL", "Link1.STL"):
        shutil.copy2(g29_source / "meshes" / name, object_dir / "meshes" / name)

    urdf = ET.parse(urdf_path).getroot()
    joint = urdf.find("joint[@name='joint1']")
    jpos = np.fromstring(joint.find("origin").get("xyz"), sep=" ")
    jr = Rotation.from_euler("xyz", np.fromstring(joint.find("origin").get("rpy"), sep=" "))
    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    axis /= np.linalg.norm(axis)
    # Only yaw the housing: the support plane stays horizontal. Its built-in
    # shaft inclination comes entirely from the URDF, not a hand-tuned column.
    base_r = Rotation.from_euler("z", np.pi / 2)
    # CAD underside support face is z=.019; the forward clamp hangs below the
    # table edge. The CAD origin itself is not the bottom support surface.
    base_pos = np.array([base_x, 0., table_height - .019])
    rotor_r = base_r * jr
    axis_world = rotor_r.apply(axis)
    frame_y = np.array([0., 1., 0.])
    frame_y -= axis_world * np.dot(frame_y, axis_world)
    frame_y /= np.linalg.norm(frame_y)
    frame = np.column_stack([axis_world, frame_y, np.cross(axis_world, frame_y)])
    visual_mesh = trimesh.load(object_dir / "meshes/Link1.STL", process=True)
    pieces = wheel_collision(visual_mesh, axis, collision_dir)

    root = ET.Element("mujoco", model="humanlike_drive_logitech_g29_v1")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", balanceinertia="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast",
                  solver="Newton", iterations="80", gravity="0 0 -9.81")
    ET.SubElement(root, "size", nconmax="2500", njmax="10000")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.15", armature="0.002")
    ET.SubElement(default, "geom", friction="1.0 .01 .001", condim="4",
                  solref=".008 1", solimp=".95 .99 .001", margin="0.0001")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1200", azimuth="135", elevation="-20")
    ET.SubElement(visual, "headlight", ambient=".35 .35 .35", diffuse=".65 .65 .65", specular=".15 .15 .15")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(root, "statistic", center=".3 0 1.0", extent="1.4")
    asset = ET.SubElement(root, "asset")
    for assets in source.findall("asset"):
        for m in assets.findall("mesh"):
            mesh = copy.deepcopy(m)
            mesh.set("file", "../robots/h1_2/meshes/" + mesh.get("file"))
            asset.append(mesh)
    for name, filename in [("g29_housing", "base_link_wheel.STL"), ("g29_visual", "Link1.STL")]:
        ET.SubElement(asset, "mesh", name=name, file="../objects/logitech_g29/meshes/" + filename)
    for path, _ in pieces:
        ET.SubElement(asset, "mesh", name="g29_" + path.stem,
                      file="../objects/logitech_g29/collision/" + path.name)
    ET.SubElement(asset, "texture", name="floor_grid", type="2d", builtin="checker",
                  rgb1=".20 .23 .27", rgb2=".24 .27 .31", width="256", height="256")
    ET.SubElement(asset, "material", name="floor_material", texture="floor_grid", texrepeat="8 8", texuniform="true")
    ET.SubElement(asset, "texture", name="studio_sky", type="skybox", builtin="gradient",
                  rgb1=".18 .22 .28", rgb2=".045 .055 .07", width="512", height="3072")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="-1 -1 3.8", dir=".2 .2 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "light", name="fill", pos="1.5 2 3", dir="-.2 -.4 -1", diffuse=".5 .5 .5")
    ET.SubElement(world, "geom", name="floor", type="plane", size="4 4 .05", material="floor_material",
                  contype="4", conaffinity="2")
    robot = copy.deepcopy(source.find("worldbody/body[@name='pelvis']"))
    # Seated lower body is baked from the vendor kinematics, not copied from an
    # old scene or grasp snapshot. Arm/hand kinematics and limits stay original.
    robot.set("pos", "0 0 .83")
    for body in robot.iter("body"):
        for j in list(body.findall("joint")):
            if j.get("name") not in ARM_NAMES and not j.get("name", "").startswith(("L_", "R_")):
                angle = -np.pi / 2 if "hip_pitch" in j.get("name", "") else np.pi / 2 if "knee" in j.get("name", "") else 0.
                old = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
                rotation = Rotation.from_quat(old[[1,2,3,0]])
                body.set("quat", quat(rotation * Rotation.from_rotvec(np.fromstring(j.get("axis", "0 0 1"), sep=" ") * angle)))
                body.remove(j)
            elif j.get("name", "").startswith(("L_", "R_")):
                j.set("damping", ".025")
                j.set("armature", ".0002")
        for k, geom in enumerate(body.findall("geom")):
            if geom.get("mesh") == "logo_link":
                body.remove(geom)
                continue
            geom.set("name", body.get("name") + f"_geom_{k}")
            is_visual = geom.get("contype") == "0"
            geom.set("group", "1" if is_visual else "3")
            geom.set("rgba", ".66 .71 .77 1" if is_visual else ".2 .65 1 .32")
            if not is_visual:
                geom.set("contype", "2")
                geom.set("conaffinity", "5")
                geom.set("mass", "0")
        name = body.get("name", "")
        if name.endswith("wrist_yaw_link"):
            side = name.split("_")[0]
            ET.SubElement(body, "site", name=f"{side}_grip", pos=".19 0 .008",
                          size=".003", rgba=".2 .9 .4 0")
    world.append(robot)

    bench = ET.SubElement(world, "body", name="workbench")
    def box(name, pos, size, color):
        return ET.SubElement(bench, "geom", name=name, type="box", pos=vec(pos), size=vec(size),
                             rgba=color, contype="4", conaffinity="2")
    # Front edge clears the wheel rim; the stock clamp can hang below the top.
    box("table_top", [base_x + .166, 0, table_height - .025], [.184, .48, .025], ".085 .10 .12 1")
    for x in (base_x + .012, base_x + .28):
        for y in (-.39, .39):
            box(f"table_leg_{x:.2f}_{y}", [x, y, (table_height - .05) / 2],
                [.025, .025, (table_height - .05) / 2], ".18 .21 .25 1")
    box("table_rear_bar", [base_x + .28, 0, .20], [.026, .415, .026], ".18 .21 .25 1")
    from humanlike_drive.env.cabin import add_cabin
    add_cabin(world, asset, PROJECT / "assets/scenes/textures", vec, quat)
    housing = ET.SubElement(world, "body", name="g29_housing", pos=vec(base_pos), quat=quat(base_r))
    ET.SubElement(housing, "geom", name="housing_visual", type="mesh", mesh="g29_housing",
                  rgba=".12 .14 .17 1", contype="0", conaffinity="0", group="1")
    # Housing is outside grasp region; its conservative hull is separately shown
    # in collision view. The rotating wheel never uses its full-mesh convex hull.
    ET.SubElement(housing, "geom", name="housing_collision", type="mesh", mesh="g29_housing",
                  rgba=".4 .5 .6 .25", contype="4", conaffinity="2", group="3")
    rotor = ET.SubElement(housing, "body", name="g29_wheel", pos=vec(jpos), quat=quat(jr))
    inertial = urdf.find("link[@name='Link1']/inertial")
    inertia = inertial.find("inertia")
    ET.SubElement(rotor, "inertial", pos=inertial.find("origin").get("xyz"),
                  mass=inertial.find("mass").get("value"),
                  fullinertia=vec([float(inertia.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]))
    ET.SubElement(rotor, "joint", name="wheel_joint", axis=vec(axis), type="hinge",
                  limited="true", range=vec(np.deg2rad([-450, 450])), damping=".02", frictionloss=".015")
    ET.SubElement(rotor, "geom", name="wheel_visual", type="mesh", mesh="g29_visual",
                  rgba=".13 .15 .19 1", contype="0", conaffinity="0", group="1", mass="0")
    for path, label in pieces:
        ET.SubElement(rotor, "geom", name="collision_" + path.stem, type="mesh", mesh="g29_" + path.stem,
                      contype="1", conaffinity="2", group="3", mass="0",
                      rgba=".15 .95 .48 1" if label == "rim" else ".95 .20 .75 1" if label == "paddle" else ".95 .65 .15 1")
    local_frame = rotor_r.inv().as_matrix() @ frame
    ET.SubElement(rotor, "site", name="wheel_frame", quat=quat(Rotation.from_matrix(local_frame)),
                  type="sphere", size=".003", rgba="0 0 0 0")

    eq = ET.SubElement(root, "equality")
    actuators = ET.SubElement(root, "actuator")
    for name in ARM_NAMES:
        i = ARM_SUFFIXES.index(name.removeprefix("left_").removeprefix("right_").removesuffix("_joint"))
        ET.SubElement(actuators, "position", name=name, joint=name, kp=str([360, 360, 240, 240, 90, 90, 90][i]),
                      kv=str(np.sqrt(2) * [15, 15, 10, 10, 4, 4, 4][i]), forcerange=vec([-40, 40] if i < 2 else [-18, 18]))
    # Use the matching H1-2 URDF; other Inspire variants have different ratios.
    for relation in relations:
        ET.SubElement(eq, "joint", name="mimic_" + relation["joint"], joint1=relation["joint"],
                      joint2=relation["parent"], polycoef=vec([relation["offset"], relation["multiplier"], 0, 0, 0]),
                      solref=".004 1")
    ranges = {j.get("name"): np.fromstring(j.get("range"), sep=" ") for j in robot.iter("joint")}
    for relation in reversed(relations):
        bounds = np.sort((ranges[relation["joint"]] - relation["offset"]) / relation["multiplier"])
        parent = relation["parent"]
        ranges[parent] = np.array([max(ranges[parent][0], bounds[0]), min(ranges[parent][1], bounds[1])])
    for side in ("L", "R"):
        for suffix in HAND_SUFFIXES:
            name = f"{side}_{suffix}_joint"
            ET.SubElement(actuators, "position", name=name, joint=name, kp="1.5", kv=".06", forcerange="-.35 .35",
                          ctrllimited="true", ctrlrange=vec(ranges[name]))
    center = base_pos + base_r.apply(jpos)
    camera(world, "overview", [.65, -3.10, 2.12], [.55, 0, 1.0], 48)
    camera(world, "cabin", [-1.25, -2.1, 1.95], [.48, 0, 1.0], 52)
    camera(world, "side", [.3, -1.3, 1.4], [.38, 0, 1.05], 50)
    camera(world, "hands", center + [-.17, -.48, .40], center + [-.025, 0, 0], 46)
    camera(world, "driver", center + [-.38, 0, .32], center, 46)
    camera(world, "driver_eye", [.18, 0, 1.47], [1.58, 0, 1.26], 72)
    camera(world, "wheel", center + [-.48, -.22, .22], center, 43)
    camera(world, "grip_side", center + frame @ [-.07, .38, .055],
           center + frame @ [0., .125, 0.], 38)
    camera(world, "grip_rear", center + frame @ [.17, .30, .075],
           center + frame @ [.005, .125, 0.], 42)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(SCENE, encoding="utf-8", xml_declaration=True)
    report = {
        "robot_source": str(robot_source), "robot_sha256": sha(robot_source),
        "hand_mimic_source": str(hand_urdf), "hand_urdf_sha256": sha(hand_urdf),
        "hand_mimic_relations": list(relations),
        "g29_source": str(urdf_path), "g29_urdf_sha256": sha(urdf_path),
        "mesh_sha256": {p.name: sha(p) for p in (object_dir / "meshes").glob("*.STL")},
        "base_rotation": "world yaw +90deg only; CAD support plane remains horizontal",
        "robot_pose": "fixed seated pelvis at z=.83; hip pitch -90deg / knee +90deg baked into vendor transforms",
        "base_position_m": base_pos.tolist(), "wheel_origin_world_m": center.tolist(),
        "table_top_height_m": table_height,
        "cad_support_plane_local_z_m": .019,
        "wheel_axis_world": axis_world.tolist(),
        "shaft_inclination_deg": float(np.rad2deg(np.arcsin(abs(axis_world[2])))),
        "collision_parts": {label: sum(l == label for _, l in pieces) for label in ("rim", "paddle", "inner", "hub")},
        "wheel_limits_deg": [-450, 450],
        "limitations": ["URDF base_link.STL missing; installed base_link_wheel.STL used",
                        "original collision filenamse/bae_link.STL invalid; not imported",
                        "housing collision is conservative convex hull",
                        "robot self-collision response disabled for this fixed-seat inspection scene; external contacts active",
                        "cockpit is procedural; screens show static preview images, not a driving simulator feed",
                        "no old scene, checkpoint, grasp state or dataset loaded",
                        "damping, friction and actuator gains are provisional, not real calibration"],
    }
    (object_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return SCENE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-source", type=Path, default=ROBOT_SOURCE)
    parser.add_argument("--g29-source", type=Path, default=G29_SOURCE)
    parser.add_argument("--table-height", type=float, default=.94)
    parser.add_argument("--base-x", type=float, default=.64)
    args = parser.parse_args()
    print(build(**vars(args)))
