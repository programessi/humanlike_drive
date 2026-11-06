"""Mimic relations for the exact Inspire model attached to H1-2.

The linear relations are taken from its URDF, not tuned for a desired grasp.
They remain a model approximation until checked against the physical hand.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

HAND_URDF = Path(__file__).resolve().parents[2] / "assets/robots/h1_2/source.urdf"


@lru_cache(maxsize=4)
def mimic_relations(path=HAND_URDF):
    root = ET.parse(path).getroot()
    relations = []
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if joint.get("name", "").startswith(("L_", "R_")) and mimic is not None:
            relations.append({"joint": joint.get("name"), "parent": mimic.get("joint"),
                              "multiplier": float(mimic.get("multiplier", "1")),
                              "offset": float(mimic.get("offset", "0"))})
    if not relations:
        raise ValueError(f"No Inspire mimic definitions in {path}")
    return tuple(relations)


def effective_joint_ranges(model, relations=None):
    """Intersect driver ranges with every dependent joint's range."""
    relations = mimic_relations() if relations is None else relations
    ranges = model.jnt_range.copy()
    for _ in range(len(relations)):
        for r in reversed(relations):
            child = model.joint(r["joint"]).id
            parent = model.joint(r["parent"]).id
            if r["multiplier"] == 0:
                continue
            bounds = np.sort((ranges[child] - r["offset"]) / r["multiplier"])
            ranges[parent, 0] = max(ranges[parent, 0], bounds[0])
            ranges[parent, 1] = min(ranges[parent, 1], bounds[1])
            if ranges[parent, 0] > ranges[parent, 1]:
                raise ValueError(f"Inconsistent mimic limits for {r['parent']}")
    return ranges


def apply_mimic(model, data, side):
    for r in mimic_relations():
        if r["joint"].startswith(side + "_"):
            child = model.jnt_qposadr[model.joint(r["joint"]).id]
            parent = model.jnt_qposadr[model.joint(r["parent"]).id]
            data.qpos[child] = r["offset"] + r["multiplier"] * data.qpos[parent]


def mimic_errors(model, data):
    result = {}
    for r in mimic_relations():
        child = data.qpos[model.jnt_qposadr[model.joint(r["joint"]).id]]
        parent = data.qpos[model.jnt_qposadr[model.joint(r["parent"]).id]]
        result[r["joint"]] = float(child - r["offset"] - r["multiplier"] * parent)
    return result
