"""Offline geometry diagnostic only; never imported by training or execution.

Rotate the two initial wrist poses rigidly with the wheel, and solve each
arm along the path. This checks one possible kinematic path, not dynamic
grasp stability, contact forces, or a learned controller.
"""
import argparse
import copy
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from unified_steering.audit import digest, write_json
from unified_steering.env import SteeringEnv
from unified_steering.train import resolved_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amplitude-deg', type=float, default=20.)
    args = parser.parse_args()
    env = SteeringEnv(copy.deepcopy(resolved_config(args.config)['environment']))
    env.reset(seed=7, options={'support': 'both'})
    s = env.sim
    m, d = s.model, mujoco.MjData(s.model)
    initial_q = s.data.qpos.copy()
    positions = s.data.site_xpos[s.grips].copy()
    rotations = s.data.site_xmat[s.grips].reshape(2, 3, 3).copy()
    axis = s.data.xaxis[s.wheel_id].copy()
    rows = []
    for sign in (1, -1):
        d.qpos[:] = initial_q
        for offset in np.linspace(0., sign*args.amplitude_deg, 21):
            rotation = Rotation.from_rotvec(axis*np.deg2rad(offset)).as_matrix()
            d.qpos[s.wheel_q] = initial_q[s.wheel_q]+np.deg2rad(offset)
            errors = []
            for side in range(2):
                indices = s.qidx[7*side:7*(side+1)]
                seed = d.qpos[indices].copy()
                target_position = s.center+rotation@(positions[side]-s.center)
                target_rotation = rotation@rotations[side]

                def residual(q):
                    d.qpos[indices] = q
                    mujoco.mj_kinematics(m, d)
                    current = d.site_xmat[s.grips[side]].reshape(3, 3)
                    return np.r_[d.site_xpos[s.grips[side]]-target_position,
                                 .12*Rotation.from_matrix(target_rotation@current.T).as_rotvec(),
                                 .0002*(q-seed)]

                bounds = env.executor.bounds[7*side:7*(side+1)]
                solution = least_squares(residual, np.clip(seed, *bounds.T), bounds=bounds.T,
                                         max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                error = residual(solution.x)
                errors.append(dict(solver_success=bool(solution.success),
                                   position_error_m=float(np.linalg.norm(error[:3])),
                                   rotation_error_deg=float(np.rad2deg(np.linalg.norm(error[3:6])/.12))))
            mujoco.mj_forward(m, d)
            limits = m.jnt_range[s.jids[:14]]
            q = d.qpos[s.qidx[:14]]
            rows.append(dict(offset_deg=float(offset), arms=errors,
                             min_arm_margin_deg=float(np.rad2deg(np.minimum(q-limits[:, 0], limits[:, 1]-q).min())),
                             min_other_geometry_gap_m=min(s.clearance.minimum(d, pairs) for pairs in s.clearance.other)))
    passed = all(all(e['solver_success'] and e['position_error_m']<=.002 and e['rotation_error_deg']<=.5
                     for e in r['arms']) and r['min_arm_margin_deg']>=2.-1e-6
                 and r['min_other_geometry_gap_m']>=0. for r in rows)
    report = dict(kind='offline_rigid_grasp_kinematic_path_not_dynamic_validation',
                  passed=passed, config_sha256=digest(args.config), amplitude_deg=args.amplitude_deg,
                  max_position_error_m=max(e['position_error_m'] for r in rows for e in r['arms']),
                  max_rotation_error_deg=max(e['rotation_error_deg'] for r in rows for e in r['arms']),
                  min_arm_margin_deg=min(r['min_arm_margin_deg'] for r in rows),
                  min_other_geometry_gap_m=min(r['min_other_geometry_gap_m'] for r in rows), rows=rows)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print({k: v for k, v in report.items() if k!='rows'})
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
