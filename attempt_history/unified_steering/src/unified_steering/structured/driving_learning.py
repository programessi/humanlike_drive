"""Driving BC/DAgger and script-free evaluation, with immutable run directories."""
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .driving_task import DrivingConfig, DrivingEnv
from .legacy.regrasp.controller import RegraspControlConfig
from .demonstrator import DemonstratorConfig
from . import driving_roll_integral_audit as roll
from .evaluate import load_teacher
from .bootstrap import fit


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def init_worker(model_path, protocol, output, collect, beta):
    global _model, _protocol, _output, _collect, _beta, _expert_class
    torch.set_num_threads(1)
    _model = load_teacher(model_path) if model_path else None
    _protocol, _output, _collect, _beta = protocol, Path(output), collect, beta
    roll._TABLE = protocol['tangent_geometry']
    roll.TangentRollDemonstrator.phase_lead_deg = protocol.get('phase_lead_deg', 30.)
    _expert_class = roll.TangentRollDemonstrator
    if protocol.get('expert_type')=='shuffle':
        from .driving_shuffle_expert import ShuffleDemonstrator
        _expert_class=ShuffleDemonstrator
        for key,value in protocol.get('shuffle_parameters',{}).items():setattr(_expert_class,key,value)
    if protocol.get('expert_type')=='handoff':
        from .driving_handoff_expert import HandoffDemonstrator
        _expert_class=HandoffDemonstrator
        for key,value in protocol.get('handoff_parameters',{}).items():setattr(_expert_class,key,value)
    if protocol.get('expert_type')=='local_roll':
        from .driving_local_expert import LocalRollDemonstrator
        _expert_class=LocalRollDemonstrator
    if protocol.get('expert_type') in ('halfturn','local_halfturn'):
        from .driving_halfturn_expert import HalfturnDemonstrator
        _expert_class = HalfturnDemonstrator
        if protocol.get('expert_type')=='local_halfturn':
            from .driving_local_expert import LocalHalfturnDemonstrator
            _expert_class=LocalHalfturnDemonstrator
        for key,value in protocol.get('halfturn_parameters', {}).items():
            setattr(_expert_class,key,value)


def run_job(case):
    amplitude, direction, template, seed, *extra = case
    options = dict(free_side=-1, exact_curriculum=True, direction=direction, template=template)
    if extra: options.update(extra[0])
    cfg = DrivingConfig(amplitude_deg=amplitude,command_range_observation=_protocol.get('command_range_observation',False),teacher_contact_observation=_protocol.get('teacher_contact_observation',False), controller=RegraspControlConfig(**_protocol.get('controller',{})),
        target_speed_deg=_protocol.get('speed_deg_s', 5.),
        target_acceleration_deg_s2=_protocol.get('acceleration_deg_s2', 40.),
        arm_command_margin_deg=_protocol.get('arm_command_margin_deg',3.),
        guard_elbows_against_wheel=_protocol.get('guard_elbows_against_wheel',False),
        guard_opposite_hand_meshes=_protocol.get('guard_opposite_hand_meshes',False),
        guard_full_physical_pairs=_protocol.get('guard_full_physical_pairs',False),
        retain_preparation_hand_command=_protocol.get('retain_preparation_hand_command',False),
        nullspace_joint_limit_gain=_protocol.get('nullspace_joint_limit_gain',0.),
        qp_backend=_protocol.get('qp_backend','slsqp'),
        canonical_distance_guard_order=_protocol.get('canonical_distance_guard_order',False),
        collision_guard_margin_m=_protocol.get('collision_guard_margin_m',.012),
        collision_guard_activation_m=_protocol.get('collision_guard_activation_m',.035),
        hand_leak_alpha=_protocol.get('hand_leak_alpha',.995),hand_delta_rad=_protocol.get('hand_delta_rad',.03))
    env = DrivingEnv(cfg)
    wall = time.monotonic(); rows = []; data = {k: [] for k in ['actor', 'privileged', 'action']}
    stem = f'a{amplitude:g}_d{direction:+d}_{template}_s{seed}'
    if extra:stem+='_'+hashlib.sha256(json.dumps(extra[0],sort_keys=True).encode()).hexdigest()[:10]
    try:
        obs, preparation = env.reset(seed=seed, options=options)
        if hasattr(_model,"reset"):_model.reset()
        expert = _expert_class(env, DemonstratorConfig(**_protocol['demonstrator'])) if _collect or _model is None else None
        if expert is not None:
            for key,value in options.get('expert_parameters', {}).items(): setattr(expert,key,value)
            if 'grip_preload_rad' in options:
                for ids in [slice(2,6),slice(8,12)]:
                    expert.hand_target[ids] += options['grip_preload_rad']
                expert.hand_target = np.clip(expert.hand_target,env.controller.hand_limits[:,0]+.03,env.controller.hand_limits[:,1]-.03)
            if 'roll_gain' in options: expert.roll *= options['roll_gain']
        if _collect and 'teacher_contact' in obs:data['teacher_contact']=[]
        if _collect and hasattr(expert,'ipsilateral_regrasp'):
            data.update({k:[] for k in ['pose','hand_target','phase','omega']})
        rebased_expert_legs=set()
        rng = np.random.default_rng(seed)
        terminated = truncated = False; reason = ''
        for _ in range(round(env.config.episode_seconds/env.dt)):
            if expert is not None:
                # Collection-only correction of stale demonstrator memory.
                # Never resets robot, simulator, command clock or learned actor.
                if _collect and _protocol.get('resynchronize_expert_at_zero',False):
                    for leg,segment in enumerate(env.timeline['segments']):
                        if leg < 2 or leg % 2 or leg in rebased_expert_legs or env.time+1e-8 < segment['start']:continue
                        old_events=list(expert.events);expert=_expert_class(env,DemonstratorConfig(**_protocol['demonstrator']))
                        expert.direction=int(np.sign(segment['target']-segment['initial']));expert.motion_direction=expert.direction;expert.lower=0 if expert.direction<0 else 1;expert.free=expert.lower;expert.leg_start_time=env.time
                        expert.events=old_events+[dict(event='collection_expert_memory_rebased',time=env.time,leg=leg,wheel_deg=float(np.rad2deg(env.wheel_angle)),forces_n=env.contact_forces.tolist())];rebased_expert_legs.add(leg)
                label = expert.predict(obs)[0]
                if _collect:
                    if 'teacher_contact' in data:data['teacher_contact'].append(obs['teacher_contact'].copy())
                    for key, value in [('actor', obs['actor']), ('privileged', obs['privileged']), ('action', label)]:
                        data[key].append(np.asarray(value, np.float32).copy())
                if _collect and 'pose' in data:
                    poses=[]
                    for side in range(2):
                        p,r,dp,dr=expert.intent[side];f=env.controller.axis_frame
                        poses.append(np.r_[f.T@(p-env.controller.center),(f.T@r)[:,:2].reshape(6),f.T@dp,f.T@dr])
                    data['pose'].append(np.concatenate(poses).astype(np.float32));data['hand_target'].append(expert.last_hand_target.astype(np.float32));data['phase'].append(expert.controlled_phase);data['omega'].append(expert.intent_omega)
            predicted=_model.predict(obs,deterministic=True)[0] if _model is not None else None
            if _model is None or (_collect and rng.random() < _beta):
                action = label
            else:
                action = predicted
            obs, reward, terminated, truncated, info = env.step(action)
            c = env.controller
            rows.append(dict(time=env.time, angle_deg=float(np.rad2deg(env.wheel_angle)),
                target_deg=float(np.rad2deg(env.target_angle)), error_deg=info['error_deg'],
                physics_peak_error_deg=info['physics_peak_error_deg'],
                forces_n=[info['left_force'], info['right_force']],
                torques_nm=[info['left_wheel_contact_torque_nm'], info['right_wheel_contact_torque_nm']],
                centering_nm=info['centering_torque_nm'],
                learned_memory_phase_deg=float(np.rad2deg(getattr(getattr(_model,'policy',None),'phase',np.nan))),
                learned_intent_memory=int(getattr(getattr(_model,'policy',None),'memory',-1)),
                expert_action_stage=str(getattr(expert,'action_stage','')),
                expert_action_mode=str(getattr(expert,'action_mode','')),
                expert_action_side=int(getattr(expert,'action_side',-1)),
                learned_memory_age=float(getattr(getattr(_model,'policy',None),'memory_age',0.)),
                learned_internal_rebases=int(getattr(getattr(_model,'policy',None),'internal_rebase_count',0)),
                learned_previous_intent=(getattr(getattr(_model,'policy',None),'previous_intent',None).detach().cpu().numpy() if getattr(getattr(_model,'policy',None),'previous_intent',None) is not None else np.full(42,np.nan)),
                expert_stage=str(getattr(expert,'stage','')),
                expert_mode=str(getattr(expert,'mode','')),
                expert_phase_deg=float(np.rad2deg(getattr(expert,'controlled_phase',np.nan))),
                min_arm_margin_deg=info['physics_min_arm_margin_deg'], collision_n=info['physics_collision_peak_n'],
                qp_failures=info['qp_failures'], workspace_clips=info['workspace_clips'],
                limiting_joint_names=[info['left_limiting_joint'],info['right_limiting_joint']],
                active_q=env.data.qpos[np.r_[c.arm_q,c.hand_q]].tolist(),
                slip_deg=np.rad2deg(env.slip).tolist(), reward=reward))
            reason = info['termination_reason']
            if info['qp_failures']:
                reason = reason or 'first_qp_failure'; break
            if terminated or truncated: break
        times = np.array([r['time'] for r in rows]); errors = np.array([r['error_deg'] for r in rows])
        complete = bool(not terminated and not reason and times[-1]+1e-8 >= cfg.episode_seconds)
        moving = np.zeros(len(rows), bool); stop_peaks = []; segment_errors = []
        for segment in env.timeline['segments']:
            mask = (times >= segment['start']) & (times <= segment['end'])
            moving |= mask
            segment_errors.append(float(np.sqrt(np.mean(errors[mask]**2))) if mask.any() else None)
            mask = (times >= segment['hold_end']-3) & (times <= segment['hold_end'])
            stop_peaks.append(max((r['physics_peak_error_deg'] for r, flag in zip(rows, mask) if flag), default=None) if complete else None)
        motion_rmse = float(np.sqrt(np.mean(errors[moving]**2))) if moving.any() and complete else None
        hard = bool(reason or max(r['collision_n'] for r in rows) > 1. or info['qp_failures'])
        passed = complete and not hard and motion_rmse <= 1. and max(stop_peaks) <= 1.
        joint_violations=[]
        for joint in env.limited_ids:
            value=env.data.qpos[env.model.jnt_qposadr[joint]]
            lo,hi=env.model.jnt_range[joint]
            if value<lo or value>hi:
                joint_violations.append(dict(name=env.model.joint(joint).name,value=float(value),range=[float(lo),float(hi)],
                    violation_rad=float(max(lo-value,value-hi))))
        record = dict(case=case, amplitude_deg=amplitude, direction=direction, template=template, seed=seed,
            complete=complete, passed=bool(passed), hard_failure=hard, termination_reason=reason,
            planned_seconds=cfg.episode_seconds, actual_seconds=env.time, samples=len(rows),
            full_rmse_deg=float(np.sqrt(np.mean(errors**2))) if complete else None,
            prefix_rmse_deg=float(np.sqrt(np.mean(errors**2))), motion_rmse_deg=motion_rmse,
            segment_rmse_deg=segment_errors, stop_peaks_deg=stop_peaks,
            peak_error_deg=max(r['physics_peak_error_deg'] for r in rows),
            min_arm_margin_deg=min(r['min_arm_margin_deg'] for r in rows),
            collision_peak_n=max(r['collision_n'] for r in rows), qp_failures=info['qp_failures'],
            initial_wheel_deg=float(np.rad2deg(preparation['wheel_angle'])),
            qp_diagnostic={k:getattr(c.last_qp_result,k) for k in ['reason','iterations','primal_residual','dual_residual']} if hasattr(c,'last_qp_result') else None,
            joint_violations_at_end=joint_violations, expert_events=getattr(expert,'events',None), timeline=env.timeline, expert_probability=_beta if _collect else None,
            learned_evaluation=bool(_model is not None and not _collect), wall_seconds=time.monotonic()-wall)
        np.savez_compressed(_output/'traces'/f'{stem}.npz', **{k: np.array([r[k] for r in rows]) for k in rows[0]})
        if _collect:
            np.savez_compressed(_output/'labels'/f'{stem}.npz', **{k: np.array(v) for k, v in data.items()})
        write(_output/'cases'/f'{stem}.json', record)
        print('CASE '+json.dumps(record), flush=True)
        return record
    finally:
        env.close()


def run_suite(args, collect):
    protocol = json.loads(args.protocol.read_text())
    if args.model:
        torch.set_num_threads(1)
        preflight=load_teacher(args.model);del preflight
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['cases', 'traces']+(['labels'] if collect else []): (args.output/name).mkdir()
    shutil.copy2(args.protocol, args.output/'protocol.json')
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None
    metadata = dict(model=str(args.model) if args.model else None, sha256=model_hash,
        collection=collect, script_at_learned_evaluation=False, independent=args.independent,
        phase_lead_deg=protocol.get('phase_lead_deg', 30.), interface=protocol.get('interface','driving-v1-v4-executor'),
        task=asdict(DrivingConfig(command_range_observation=protocol.get('command_range_observation',False),teacher_contact_observation=protocol.get('teacher_contact_observation',False),controller=RegraspControlConfig(**protocol.get('controller',{})),arm_command_margin_deg=protocol.get('arm_command_margin_deg',3.),
            guard_elbows_against_wheel=protocol.get('guard_elbows_against_wheel',False),
            guard_opposite_hand_meshes=protocol.get('guard_opposite_hand_meshes',False),
            guard_full_physical_pairs=protocol.get('guard_full_physical_pairs',False),
            retain_preparation_hand_command=protocol.get('retain_preparation_hand_command',False),
            nullspace_joint_limit_gain=protocol.get('nullspace_joint_limit_gain',0.),
            qp_backend=protocol.get('qp_backend','slsqp'),
            canonical_distance_guard_order=protocol.get('canonical_distance_guard_order',False),
            collision_guard_margin_m=protocol.get('collision_guard_margin_m',.012),
            collision_guard_activation_m=protocol.get('collision_guard_activation_m',.035),
            hand_leak_alpha=protocol.get('hand_leak_alpha',.995),hand_delta_rad=protocol.get('hand_delta_rad',.03))), started_unix=time.time())
    write(args.output/'metadata.json', metadata)
    source_dir = args.output/'source_snapshot'; source_dir.mkdir()
    for name in ['driving_task.py', 'driving_learning.py', 'driving_roll_integral_audit.py', 'driving_halfturn_expert.py', 'driving_local_expert.py', 'driving_policy.py', 'driving_handoff_expert.py', 'box_qp.py', 'driving_memory.py', 'driving_recurrent.py', 'driving_shuffle_expert.py', 'driving_pose_policy.py', 'driving_coordination_policy.py', 'driving_intent_state_policy.py', 'driving_manifold_policy.py', 'driving_conditional_memory.py', 'driving_kernel_memory.py', 'driving_release_memory.py', 'driving_history_intent.py', 'driving_local_kernel.py', 'driving_contact_history.py', 'driving_factor_transition.py', 'driving_path_basis.py', 'driving_frame.py', 'driving_anchor_gate.py', 'driving_command_gate.py', 'evaluate.py', 'driving_hand_fit.py']:
        shutil.copy2(Path(__file__).with_name(name), source_dir/name)
    rows = []
    with mp.get_context('fork').Pool(args.workers, initializer=init_worker,
            initargs=(args.model, protocol, args.output, collect, args.beta)) as pool:
        for record in pool.imap(run_job, protocol['cases']):
            rows.append(record)
            direction_pass = {str(d): float(np.mean([r['passed'] for r in rows if r['direction'] == d])) for d in (-1, 1) if any(r['direction'] == d for r in rows)}
            passed = len(rows) == len(protocol['cases']) and np.mean([r['passed'] for r in rows]) >= .95 and min(direction_pass.values()) >= .9 and not any(r['hard_failure'] for r in rows)
            result = dict(cases=rows, passed=bool(passed), count=len(rows), passes=sum(r['passed'] for r in rows),
                direction_pass_fraction=direction_pass, learned_evaluation=bool(args.model and not collect), independent=args.independent)
            write(args.output/'summary.json', result)
    if args.model:
        assert hashlib.sha256(args.model.read_bytes()).hexdigest() == model_hash
    print('SUITE '+json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)


def fit_driving(model, dataset, protocol):
    if not protocol.get('wrist_loss_weight'):
        return fit(model,dataset,protocol['epochs'],protocol)
    arrays = {key:torch.from_numpy(value) for key,value in dataset.items()}
    parameters = list(model.policy.mlp_extractor.actor.parameters())+list(model.policy.action_net.parameters())
    if not protocol.get('freeze_encoder',False):
        parameters += list(model.policy.features_extractor.parameters())
    optimizer = torch.optim.Adam(parameters,lr=protocol['bc_learning_rate'])
    weights = torch.tensor([protocol['wrist_loss_weight']]*12+[1.]*12)
    sample_weights = torch.where(arrays['actor'][:,128].abs()<1e-5,
        float(protocol.get('hold_loss_weight',1.)),1.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs'],eta_min=protocol.get('final_learning_rate',protocol['bc_learning_rate']))
    losses=[];n=len(arrays['actor']);model.policy.set_training_mode(True)
    for epoch in range(protocol['epochs']):
        total=0.
        for ids in torch.randperm(n).split(protocol['bc_batch_size']):
            obs={key:arrays[key][ids] for key in ['actor','privileged']}
            mean=model.policy.get_distribution(obs).distribution.mean
            loss=(((mean-arrays['action'][ids]).square()*weights).mean(-1)*sample_weights[ids]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            total+=loss.item()*len(ids)
        losses.append(total/n);scheduler.step()
        if epoch%10==0 or epoch==protocol['epochs']-1:
            print(json.dumps(dict(bc_epoch=epoch,samples=n,weighted_loss=losses[-1])),flush=True)
    model.policy.set_training_mode(False)
    model.policy.optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-5,eps=1e-5)
    return losses


def train(args):
    protocol = json.loads(args.protocol.read_text()); torch.set_num_threads(1)
    torch.manual_seed(protocol['seed']); np.random.seed(protocol['seed'])
    args.output.mkdir(parents=True, exist_ok=False); shutil.copy2(args.protocol, args.output/'protocol.json')
    model = load_teacher(args.model)
    metadata = dict(parent=str(args.model), parent_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        interface=protocol.get('interface','driving-v1-v4-executor'), training_method='BC-DAgger', critic_updated=False,
        task=asdict(DrivingConfig(controller=RegraspControlConfig(**protocol.get('controller',{})),arm_command_margin_deg=protocol.get('arm_command_margin_deg',3.),
            guard_elbows_against_wheel=protocol.get('guard_elbows_against_wheel',False),
            guard_opposite_hand_meshes=protocol.get('guard_opposite_hand_meshes',False),
            guard_full_physical_pairs=protocol.get('guard_full_physical_pairs',False),
            retain_preparation_hand_command=protocol.get('retain_preparation_hand_command',False),
            nullspace_joint_limit_gain=protocol.get('nullspace_joint_limit_gain',0.),
            qp_backend=protocol.get('qp_backend','slsqp'),
            canonical_distance_guard_order=protocol.get('canonical_distance_guard_order',False),
            collision_guard_margin_m=protocol.get('collision_guard_margin_m',.012),
            collision_guard_activation_m=protocol.get('collision_guard_activation_m',.035),
            hand_leak_alpha=protocol.get('hand_leak_alpha',.995),hand_delta_rad=protocol.get('hand_delta_rad',.03))),
        status='training', script_at_evaluation=False)
    write(args.output/'config.json', metadata)
    parts = {k: [] for k in ['actor', 'privileged', 'action']}; sources = []
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            z = np.load(path)
            for key in parts: parts[key].append(z[key])
            sources.append(dict(path=str(path), samples=len(z['actor'])))
    dataset = {key: np.concatenate(value) for key, value in parts.items()}
    if protocol.get('retention_samples', 0):
        z = np.load(protocol['retention_dataset'])
        ids = np.random.default_rng(protocol['seed']).choice(len(z['actor']), protocol['retention_samples'], replace=False)
        dataset = {key: np.concatenate([array, z[key][ids]]) for key, array in dataset.items()}
        sources.append(dict(path=protocol['retention_dataset'], samples=len(ids)))
    write(args.output/'data_sources.json', sources)
    if protocol.get('normalize_inputs'):
        from .driving_policy import normalize_from_dataset
        normalize_from_dataset(model,dataset)
        metadata['policy_variant']='normalized-driving-v1'
        write(args.output/'config.json',metadata)
    losses = fit_driving(model, dataset, protocol)
    model.save(args.output/'teacher')
    metadata.update(status='trained_requires_development', samples=len(dataset['actor']), final_loss=losses[-1],
                    model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest())
    write(args.output/'config.json', metadata); write(args.output/'losses.json', losses)
    print('TRAIN '+json.dumps(metadata), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['collect', 'evaluate', 'train'])
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--beta', type=float, default=1.); p.add_argument('--independent', action='store_true')
    p.add_argument('--data', type=Path, nargs='*', default=[])
    args = p.parse_args()
    if args.command == 'train': train(args)
    else: run_suite(args, args.command == 'collect')


if __name__ == '__main__': main()
