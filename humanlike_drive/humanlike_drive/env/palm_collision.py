"""Vendor palm CAD clipped into small convex collision columns.

Coordinates and visual CAD are unchanged. Only the collision approximation is
replaced; column width is an approximation setting, not a geometric error bound.
Actual loaded contacts must be checked against the original vendor surface.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from humanlike_drive.env.logitech_scene import PROJECT, SCENE
from humanlike_drive.env.p0_scene import build_scene

ASSETS = PROJECT / "assets/robots/h1_2/palm_collision_v1"


def build_palm_scene(path, wheel_contacts=True):
    path = build_scene(path, wheel_contacts=wheel_contacts)
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    tree = ET.parse(path)
    root = tree.getroot()
    asset = root.find("asset")
    for side, short in [("left", "L"), ("right", "R")]:
        body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        old = body.find(f"geom[@name='{side}_wrist_yaw_link_geom_2']")
        for i, piece in enumerate(manifest["pieces"][short]):
            name = f"{side}_palm_collision_{i:03d}"
            file = (ASSETS / piece).resolve()
            ET.SubElement(asset, "mesh", name=name, file=str(file))
            geom = copy.deepcopy(old)
            geom.set("name", name)
            geom.set("mesh", name)
            geom.set("pos", "0 0 0")
            geom.set("quat", "1 0 0 0")
            body.append(geom)
        body.remove(old)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def clip(mesh, origin, normal):
    verts, faces, _ = trimesh.intersections.slice_faces_plane(
        mesh.vertices, mesh.faces, np.asarray(normal, float), np.asarray(origin, float)
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def generate(cell_mm=5.0):
    ASSETS.mkdir(parents=True, exist_ok=True)
    if (ASSETS / "manifest.json").exists():
        raise FileExistsError(ASSETS / "manifest.json")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    manifest = dict(
        generator="exact triangle clipping in palm-body X/Z columns, then convex hull per column",
        cell_mm=cell_mm,
        coordinates="wrist_yaw_link body coordinates, meters",
        pieces={},
        sources={},
        notes="No extrusion, new pads, altered visuals or inertia. Convex columns may fill sealed interior cavities. Grid size is not a surface error guarantee; independent actual-contact CAD audit is required.",
    )
    for short, side in [("L", "left"), ("R", "right")]:
        g = model.geom(side + "_wrist_yaw_link_geom_1").id
        mid = model.geom_dataid[g]
        bid = model.geom_bodyid[g]
        va = model.mesh_vertadr[mid]
        vn = model.mesh_vertnum[mid]
        fa = model.mesh_faceadr[mid]
        fn = model.mesh_facenum[mid]
        verts = (
            model.mesh_vert[va : va + vn] @ data.geom_xmat[g].reshape(3, 3).T
            + data.geom_xpos[g]
            - data.xpos[bid]
        ) @ data.xmat[bid].reshape(3, 3)
        mesh = trimesh.Trimesh(
            vertices=verts, faces=model.mesh_face[fa : fa + fn], process=False
        )
        paths = []
        cell = cell_mm / 1000.0
        lo = mesh.bounds[0]
        hi = mesh.bounds[1]
        skipped = 0
        for x in np.arange(lo[0], hi[0], cell):
            slab = clip(
                clip(mesh, [x, 0, 0], [1, 0, 0]),
                [min(x + cell, hi[0]), 0, 0],
                [-1, 0, 0],
            )
            if not len(slab.vertices):
                continue
            for z in np.arange(lo[2], hi[2], cell):
                part = clip(
                    clip(slab, [0, 0, z], [0, 0, 1]),
                    [0, 0, min(z + cell, hi[2])],
                    [0, 0, -1],
                )
                points = np.unique(np.round(part.vertices, 10), axis=0)
                if (
                    len(points) < 4
                    or np.linalg.svd(points - points.mean(0), compute_uv=False)[-1]
                    < 1e-9
                ):
                    skipped += 1
                    continue
                hull = ConvexHull(points)
                piece = trimesh.Trimesh(
                    vertices=points, faces=hull.simplices, process=True
                )
                trimesh.repair.fix_normals(piece)
                p = ASSETS / f"{short}_{len(paths):03d}.stl"
                piece.export(p)
                paths.append(p.name)
        source = PROJECT / f"assets/robots/h1_2/meshes/{short}_hand_base_link.STL"
        manifest["sources"][str(source)] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        manifest["pieces"][short] = paths
        manifest.setdefault("zero_volume_cells_skipped", {})[short] = skipped
        print(
            short,
            "convex columns",
            len(paths),
            "zero-volume skipped",
            skipped,
            flush=True,
        )
    manifest["sha256"] = {
        p: hashlib.sha256((ASSETS / p).read_bytes()).hexdigest()
        for group in manifest["pieces"].values()
        for p in group
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell-mm", type=float, default=5.0)
    args = p.parse_args()
    generate(args.cell_mm)
