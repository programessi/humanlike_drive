"""Open-loop hand-only feasibility audit, explicitly not an RL result."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from wheel_manipulation.hand_only import INTERFACE_VERSION
from wheel_manipulation.hand_only.env import HandOnlyEnv, HandOnlyConfig
from wheel_manipulation.hand_only.reachability import ReachabilityConfig


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for name, sign, contact in [('hold', 0, True), ('positive', 1, True),
                                ('negative', -1, True), ('no_contact_positive', 1, False)]:
        config = HandOnlyConfig(target_mode='hold', episode_seconds=6, no_contact=not contact,
                                reachability=ReachabilityConfig(enabled=not args.no_arm_guard))
        env = HandOnlyEnv(config)
        writer = None
        rows = []
        try:
            _, prepared = env.reset(seed=7)
            if args.video and name == 'positive':
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(args.output/'scripted_positive.mp4'), fps=20,
                                            codec='libx264', macro_block_size=1)
            initial_angle = env.data.qpos[env.wheel_q]
            for step in range(120):
                moving = step < 80
                omega = sign*np.deg2rad(2) if moving else 0.
                action = np.zeros(24)
                for side in range(2):
                    local = env.frame.T @ (env.positions[side]-env.center)
                    action[side*6] = omega*np.linalg.norm(local[1:])/config.linear_speed
                    action[side*6+3] = np.rad2deg(omega)/config.angular_speed_deg
                _, _, done, _, info = env.step(action)
                reference = initial_angle + sign*np.deg2rad(2)*min(env.time, 4.)
                info.update(script_reference_deg=float(np.rad2deg(reference)),
                            script_error_deg=float(np.rad2deg(reference)-info['wheel_deg']))
                rows.append(info)
                if writer:
                    from PIL import Image, ImageDraw
                    image = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 960, 45), fill='black')
                    draw.text((8, 5), 'SCRIPTED PHYSICS AUDIT - no trained policy, H1 target guard '+str(not args.no_arm_guard), fill='yellow')
                    draw.text((8, 24), f"t={env.time:.2f}s reference={np.rad2deg(reference):.2f}deg actual={info['wheel_deg']:.2f}deg", fill='white')
                    writer.append_data(np.asarray(image))
                if done:
                    break
            with (args.output/f'{name}.csv').open('w') as file:
                output = csv.DictWriter(file, fieldnames=list(rows[0]))
                output.writeheader(); output.writerows(rows)
            summary = dict(case=name, steps=len(rows), config=asdict(config), preparation=prepared,
                           wheel_change_deg=rows[-1]['wheel_deg']-prepared['wheel_deg'],
                           scripted_rmse_deg=float(np.sqrt(np.mean([r['script_error_deg']**2 for r in rows]))),
                           both_contact_fraction=float(np.mean([r['left_contact_n']>.5 and r['right_contact_n']>.5 for r in rows])),
                           max_wrist_error_mm=max(r['max_wrist_error_mm'] for r in rows),
                           max_material_phase_drift_deg=max(r['max_material_phase_drift_deg'] for r in rows),
                           hand_collision_peak_n=max(r['hand_collision_peak_n'] for r in rows),
                           arm_clipped_steps=rows[-1]['arm_clipped_steps'],
                           arm_rejected_steps=rows[-1]['arm_rejected_steps'],
                           arm_min_joint_margin_deg=min(r['arm_min_joint_margin_deg'] for r in rows),
                           arm_min_guard_clearance_mm=min(r['arm_min_guard_clearance_mm'] for r in rows),
                           arm_min_singular_value=min(r['arm_min_singular_value'] for r in rows))
            summaries.append(summary)
            print(json.dumps({k:v for k,v in summary.items() if k not in ('config','preparation')}), flush=True)
        finally:
            if writer: writer.close()
            env.close()
    result = dict(interface=INTERFACE_VERSION, control='scripted_open_loop_no_wheel_feedback',
                  learned_policy=False, stage1_passed=False, hardware_ready=False, cases=summaries)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--no-arm-guard', action='store_true', help='Ablation only: disable H1 feasibility screening')
    main(parser.parse_args())
