"""P0 humanlike steering: a dominant hand turns, the other releases naturally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from humanlike_drive.controller.basic_grasps import apply
from humanlike_drive.controller.p0_control import RateExecutor, RoundTrip, release_motion
from humanlike_drive.controller.steering_ik import RimIK
from humanlike_drive.controller.handover_plan import HandoverPlan
from humanlike_drive.controller.p0_control import smooth_fraction
from humanlike_drive.env.logitech_scene import PROJECT, SCENE, ARM_NAMES, HAND_NAMES
from humanlike_drive.env.p0_scene import build_scene, ASSEMBLY_EXCLUSIONS
from humanlike_drive.env_runner.p0_restart import Recorder, evaluate, plots, render_video


class MotionLibrary:
    def __init__(self, model, data, output, maximum=112., step=2.):
        self.ik = RimIK(model, data, wrist_weight=.01, margin_deg=3.)
        self.attached = []
        self.retreats = {}
        self.initial_ctrl = data.ctrl.copy()
        for side in (0, 1):
            seed = self.ik.seed[side*7:side*7+7]
            points = [(0., seed.copy())]
            for sign in (-1, 1):
                warm = seed.copy()
                for degree in np.arange(step, maximum+1e-6, step):
                    warm, _ = self.ik.solve(side, np.deg2rad(sign*degree), warm)
                    points.append((np.deg2rad(sign*degree), warm.copy()))
            points.sort(key=lambda v: v[0])
            self.attached.append(PchipInterpolator([p[0] for p in points], [p[1] for p in points], axis=0))
        self.maximum = np.deg2rad(maximum)
        self.output = output

    def arm(self, side, phase, inset=0.):
        if abs(phase) > self.maximum:
            raise ValueError(f'Hand phase outside offline path: {np.rad2deg(phase)}')
        q = self.attached[side](phase).copy()
        if inset:
            from scipy.spatial.transform import Rotation
            m, d = self.ik.model, self.ik.data
            qids = self.ik.armq[side*7:side*7+7]
            d.qpos[qids] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, m.site(('left', 'right')[side]+'_grip').id)
            vids = m.jnt_dofadr[self.ik.jids[side*7:side*7+7]]
            radial = Rotation.from_rotvec(self.ik.axis*phase).apply(self.ik.anchors[side][0]-self.ik.center)
            radial -= self.ik.axis*(radial@self.ik.axis)
            delta = -inset*radial/np.linalg.norm(radial)
            jac = np.r_[jp[:, vids], .16*jr[:, vids]]
            q += np.linalg.lstsq(jac, np.r_[delta, np.zeros(3)], rcond=1e-5)[0]
            q = np.clip(q, self.ik.lower[side*7:side*7+7], self.ik.upper[side*7:side*7+7])
        return q+self.ik.preload[7*side:7*side+7]

    def retreat(self, side, start_phase, amount):
        if amount <= 1e-10:
            return self.arm(side, start_phase)
        # Interpolate a 2-degree offline grid when the wheel continues moving.
        spacing = np.deg2rad(2.)
        lower = np.floor(start_phase/spacing+1e-8)*spacing
        blend = (start_phase-lower)/spacing
        if abs(blend) > 1e-7:
            return (1.-blend)*self._retreat_grid(side, lower, amount)+blend*self._retreat_grid(side, lower+spacing, amount)
        return self._retreat_grid(side, lower, amount)

    def _retreat_grid(self, side, start_phase, amount):
        key = (side, round(float(start_phase), 8))
        if key not in self.retreats:
            points = []
            warm = self.attached[side](start_phase).copy()
            for fraction in np.linspace(0., 1., 21):
                warm, _ = self.ik.solve(side, start_phase, warm, retreat=fraction)
                points.append(warm.copy())
            self.retreats[key] = PchipInterpolator(np.linspace(0., 1., 21), points, axis=0)
        return self.retreats[key](amount)+self.ik.preload[7*side:7*side+7]

    def save(self):
        (self.output/'reachability.json').write_text(json.dumps(self.ik.records, indent=2)+'\n')


def freeze_run(output, scene, grasp_file, config):
    """Capture source and inputs before physics; old trials are never overwritten."""
    config.update(physics_dt_s=.002, control_dt_s=.05, collision_exclusions=ASSEMBLY_EXCLUSIONS,
                  load_scope='uncalibrated synthetic load, not measured G29 response',
                  arm_compensation='qfrc_bias on 14 arm DOFs, motor plus compensation checked',
                  wheel_control='passive, declared load only; no task-time qpos reset',
                  action_order=ARM_NAMES+HAND_NAMES, action='26 normalized target rates',
                  rate_limit_rad_s=[.6]*14+[1.]*12, seed=0)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'grasp_used.json').write_bytes(grasp_file.read_bytes())
    sources = list((PROJECT/'humanlike_drive').rglob('*.py'))+list((PROJECT/'humanlike_drive/config').rglob('*.json'))+[grasp_file, SCENE]
    hashes = {}
    for source in sources:
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output/'source_snapshot'/source.relative_to(PROJECT)
        destination.parent.mkdir(exist_ok=True, parents=True)
        destination.write_bytes(source.read_bytes())
    hashes[str(scene)] = hashlib.sha256(scene.read_bytes()).hexdigest()
    for asset in ET.parse(scene).getroot().findall('./asset/*'):
        if asset.get('file'):
            source = Path(asset.get('file'))
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=PROJECT)
    (output/'manifest.json').write_text(json.dumps(dict(sha256=hashes, git_commit=git.stdout.strip(),
        mujoco_version=mujoco.__version__), indent=2)+'\n')


class DominantRoundTrip:
    def __init__(self, target):
        self.reference = RoundTrip(target, initial_hold_s=5.)
        self.return_end = self.reference.duration-3.
        self.duration = self.return_end+8.
        self.dominant = 0 if target > 0 else 1
        self.other = 1-self.dominant

    def sample(self, t, closed):
        ref, phase, segment = self.reference.sample(t)
        support = [True, True]
        retreat, fingers = 0., closed.copy()
        if t < 2.:
            pass
        elif t < 5.:
            phase = 5
            retreat, fingers = release_motion(t-2., closed)
        elif t < self.return_end:
            retreat, fingers = release_motion(3., closed)
            support[self.other] = False
        elif t < self.return_end+5.:
            phase = 7
            retreat, fingers = release_motion(t-self.return_end, closed, regrasp=True)
        else:
            phase = 4
        return ref, phase, segment, support, retreat, fingers


def add_handover_evaluation(result, arrays, plan, dt):
    a = arrays
    new, old = plan.receiving, plan.original
    checks = result['checks']
    hold = a['phase'] == 10
    transfer = np.isin(a['phase'], [8, 9])
    both = transfer & (np.min(a['rim_forces'][:, new], axis=1) > .05) & (
        np.sum(a['rim_forces'][:, old], axis=1) > .05)
    signed = -np.sign(a['load_torque'])*a['hand_wheel_torque'][:, new]
    witness = both & (signed >= np.maximum(.02, .5*np.abs(a['load_torque'])))
    longest = current = 0
    for valid in witness:
        current = current+1 if valid else 0
        longest = max(longest, current)
    checks['handover_overlap_ge_0p1s'] = longest*dt >= .1
    checks['handover_new_hand_alone_10s'] = np.count_nonzero(hold)*dt >= 9.998
    checks['handover_old_hand_clear'] = bool(hold.any() and np.max(np.sum(a['rim_forces'][hold, old], axis=1)) <= .05)
    checks['handover_nonzero_load'] = bool(hold.any() and np.min(np.abs(a['load_torque'][hold])) >= .02)
    checks['handover_new_mean_counterload_torque_ge_0p02Nm'] = bool(hold.any() and np.mean(signed[hold]) >= .02)
    window = np.isin(a['phase'], [9, 10])
    if window.any():
        error = np.rad2deg(a['wheel_angle'][window]-a['reference'][window])
        retreat = float(np.max(abs(error-error[0])))
    else:
        retreat = float('inf')
    checks['handover_angle_retreat_le_2deg'] = retreat <= 2.
    if plan.moving:
        moving = np.isin(a['phase'], [7, 8, 9, 10])
        speed = np.gradient(np.rad2deg(a['reference']), a['time'])
        idx = np.flatnonzero(moving)
        inner = idx[(a['time'][idx] > plan.approach+.5) & (a['time'][idx] < plan.target_hold-.5)]
        checks['handover_reference_keeps_moving'] = bool(len(inner) and np.min(plan.sign*speed[inner]) >= .99)
    result['handover'] = dict(new_support=('left', 'right')[new], moving=plan.moving,
        overlap_s=float(longest*dt), new_mean_counterload_torque_Nm=float(np.mean(signed[hold])) if hold.any() else None,
        relative_angle_retreat_deg=retreat, receiving_grip_epoch_deg=float(np.rad2deg(plan.new_epoch)),
        final_outbound_angle_deg=float(np.rad2deg(plan.final_target)))


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'trajectory.npz').exists():
        raise FileExistsError(output)
    scene = build_scene(output/'scene.xml', wheel_contacts=not args.negative_control)
    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
    grasp_file = Path(args.grasp).resolve()
    apply(m, d, 'new', {'new': json.loads(grasp_file.read_text())})
    executor = RateExecutor(m, d)
    handover = args.case in ('handover', 'moving_handover')
    plan = HandoverPlan(args.angle, moving=args.case == 'moving_handover') if handover else DominantRoundTrip(args.angle)
    freeze_run(output, scene, grasp_file, {**vars(args), 'expected_duration_s': plan.duration,
        'method': 'offline posture-aware IK, finite-motor contact-driven dominant steering',
        'palm_regulation': {'target_N': .8, 'gain_m_per_N_s': .0012, 'inset_limit_m': .003,
                            'scope': 'simulation contact feedback; local scratch-Jacobian arm target adjustment'}})
    print('Building offline arm paths', flush=True)
    library = MotionLibrary(m, d, output)
    library.retreat(plan.receiving if handover else plan.other, 0., 1.)
    library.save()
    recorder = Recorder(m)
    armv = [m.jnt_dofadr[m.joint(n).id] for n in ARM_NAMES]
    wv = m.jnt_dofadr[m.joint('wheel_joint').id]
    wq = m.jnt_qposadr[m.joint('wheel_joint').id]
    initial_ctrl = d.ctrl.copy()
    hand_slice = None if handover else slice(14+plan.other*6, 20+plan.other*6)
    period = round(executor.dt/m.opt.timestep)
    desired = initial_ctrl.copy()
    inset = np.zeros(2)
    transfer_lead = 0.
    effective_transfer = 0.
    duration = min(plan.duration, args.duration) if args.duration else plan.duration
    for step in range(round(duration/m.opt.timestep)):
        t = float(d.time)
        if handover:
            state = plan.sample(t, initial_ctrl[14:].reshape(2, 6))
            ref, phase, segment, support = (state[k] for k in ('reference', 'phase', 'target', 'support'))
        else:
            ref, phase, segment, support, retreat, fingers = plan.sample(t, initial_ctrl[hand_slice])
        if step % period == 0:
            lead = np.clip(args.feedback_gain*(ref-d.qpos[wq]), -np.deg2rad(3.), np.deg2rad(3.))
            desired = initial_ctrl.copy()
            if handover:
                if recorder.rows and phase == 8:
                    required = -recorder.rows[-1]['load_torque']
                    measured = recorder.rows[-1]['hand_wheel_torque'][plan.receiving]
                    transfer_lead = float(np.clip(transfer_lead+.3*executor.dt*(required-measured),
                                                  -np.deg2rad(1.), np.deg2rad(1.)))
                effective_transfer = transfer_lead
                if phase == 4:
                    effective_transfer *= 1.-smooth_fraction((t-plan.return_end-5.)/3.)
                for side in (0, 1):
                    r = state['retreat'][side]
                    if recorder.rows and phase != 0 and r < .05 and np.max(abs(
                            state['fingers'][side]-initial_ctrl[14+6*side:20+6*side])) < .35:
                        palm = recorder.rows[-1]['rim_forces'][side, 4]
                        inset[side] = np.clip(inset[side]+.0012*executor.dt*(.8-palm), 0., .003)
                    if r > .1:
                        inset[side] = 0.
                    p = state['hand_phases'][side]
                    angle_correction = lead+(effective_transfer if side == plan.receiving else 0.)
                    base = library.retreat(side, p, r)
                    correction = library.arm(side, p+angle_correction, inset[side])-library.retreat(side, p, 0.)
                    desired[7*side:7*side+7] = base+(1.-r)*correction
                    desired[14+6*side:20+6*side] = state['fingers'][side]
            else:
                if recorder.rows and phase != 0:
                    palm = recorder.rows[-1]['rim_forces'][plan.dominant, 4]
                    inset[plan.dominant] = np.clip(inset[plan.dominant]+.0012*executor.dt*(.8-palm), 0., .003)
                for side in (0, 1):
                    arm_slice = slice(7*side, 7*side+7)
                    if side == plan.other and phase not in (0, 4):
                        desired[arm_slice] = library.retreat(side, 0., retreat)
                        correction = library.arm(side, ref+lead)-library.arm(side, 0.)
                        desired[arm_slice] += (1.-retreat)*correction
                        desired[hand_slice] = fingers
                    else:
                        desired[arm_slice] = library.arm(side, ref+lead, inset[side])
            d.ctrl[:] = executor.follow(desired, d.qpos[executor.qids])
        load = -args.load_k*d.qpos[wq]-args.load_b*d.qvel[wv]
        d.qfrc_applied[:] = 0.
        d.qfrc_applied[armv] = d.qfrc_bias[armv]
        d.qfrc_applied[wv] = load
        mujoco.mj_forward(m, d)
        recorder.sample(d, ref, phase, segment, executor, desired, load, support)
        recorder.rows[-1]['grip_inset_m'] = inset.copy()
        if handover:
            recorder.rows[-1]['hand_phase_rad'] = state['hand_phases'].copy()
            recorder.rows[-1]['hand_retreat_fraction'] = state['retreat'].copy()
            recorder.rows[-1]['handover_applied_arm_lead_rad'] = effective_transfer
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or np.max(np.abs(d.qvel)) > 100.:
            break
        if step % 2500 == 0:
            print(f't={t:.1f} phase={phase} ref={np.rad2deg(ref):+.1f} wheel={np.rad2deg(d.qpos[wq]):+.2f}', flush=True)
    arrays = recorder.save(output)
    library.save()
    result = evaluate(arrays, m, plan.duration, negative_control=args.negative_control,
        hold_phase_ids=[0, 6, 8, 10, 2, 4] if handover else None,
        motion_phase_ids=[1, 7, 8, 9, 10, 3] if handover and plan.moving else None)
    if handover:
        add_handover_evaluation(result, arrays, plan, m.opt.timestep)
    else:
        single = (arrays['phase'] == 2)
        result['checks']['inactive_hand_clear_of_rim_during_target_hold'] = bool(single.any() and
            np.max(np.sum(arrays['rim_forces'][single, plan.other], axis=1)) <= .05)
    result['passed'] = bool(all(result['checks'].values()) and not args.negative_control)
    (output/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    plots(output, arrays)
    print(json.dumps(result, indent=2), flush=True)
    if args.video:
        print(render_video(output, args.camera), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--grasp', default='assets/scenes/p0_clear_grasp_v1.json')
    parser.add_argument('--angle', type=float, default=90.)
    parser.add_argument('--case', choices=['dominant', 'handover', 'moving_handover'], default='dominant')
    parser.add_argument('--load-k', type=float, default=.1)
    parser.add_argument('--load-b', type=float, default=.02)
    parser.add_argument('--feedback-gain', type=float, default=1.5)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--negative-control', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--camera', default='driver')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
