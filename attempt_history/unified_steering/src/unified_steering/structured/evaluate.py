"""Frozen tracking gates shared by migration, training and independent checks."""
import argparse
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from .env import SteeringTeacherEnv, TaskConfig
from .legacy.regrasp.policy import TeacherPolicy


def load_teacher(path, env=None):
    path_meta=Path(path).parent/'config.json'
    if path_meta.exists() and json.loads(path_meta.read_text()).get('policy_variant')=='normalized-path-basis-driving-v1':
        from .driving_path_basis import PathBasisModel
        return PathBasisModel.load(path)
    factor_meta=Path(path).parent/'config.json'
    if factor_meta.exists() and json.loads(factor_meta.read_text()).get('policy_variant')=='factorized-contact-history-driving-v1':
        from .driving_factor_transition import FactorContactModel
        return FactorContactModel.load(path)
    contact_meta=Path(path).parent/'config.json'
    if contact_meta.exists() and json.loads(contact_meta.read_text()).get('policy_variant')=='privileged-contact-history-driving-v1':
        from .driving_contact_history import ContactHistoryModel
        return ContactHistoryModel.load(path)
    kernel_meta=Path(path).parent/'config.json'
    if kernel_meta.exists() and json.loads(kernel_meta.read_text()).get('policy_variant')=='local-normalized-kernel-driving-v1':
        from .driving_local_kernel import LocalKernelModel
        return LocalKernelModel.load(path)
    history_meta=Path(path).parent/'config.json'
    if history_meta.exists() and json.loads(history_meta.read_text()).get('policy_variant')=='autoregressive-history-intent-driving-v1':
        from .driving_history_intent import HistoryIntentModel
        return HistoryIntentModel.load(path)
    policy_class=TeacherPolicy
    release_meta=Path(path).parent/'config.json'
    if release_meta.exists() and json.loads(release_meta.read_text()).get('policy_variant')=='release-neural-memory-driving-v1':
        from .driving_release_memory import ReleaseMemoryModel
        return ReleaseMemoryModel.load(path)
    kernel_meta=Path(path).parent/'config.json'
    if kernel_meta.exists() and json.loads(kernel_meta.read_text()).get('policy_variant')=='kernel-neural-memory-driving-v1':
        from .driving_kernel_memory import KernelMemoryModel
        return KernelMemoryModel.load(path)
    conditional_meta=Path(path).parent/'config.json'
    if conditional_meta.exists() and json.loads(conditional_meta.read_text()).get('policy_variant')=='conditional-neural-memory-driving-v1':
        from .driving_conditional_memory import ConditionalMemoryModel
        return ConditionalMemoryModel.load(path)
    manifold_meta=Path(path).parent/'config.json'
    if manifold_meta.exists() and json.loads(manifold_meta.read_text()).get('policy_variant')=='neural-manifold-memory-driving-v1':
        from .driving_manifold_policy import ManifoldModel
        return ManifoldModel.load(path)
    neural_meta=Path(path).parent/'config.json'
    if neural_meta.exists() and json.loads(neural_meta.read_text()).get('policy_variant')=='neural-intent-memory-driving-v1':
        from .driving_intent_state_policy import IntentStateModel
        return IntentStateModel.load(path)
    coordination_meta=Path(path).parent/'config.json'
    if coordination_meta.exists() and json.loads(coordination_meta.read_text()).get('policy_variant')=='coordination-pose-driving-v1':
        from .driving_coordination_policy import CoordinationModel
        return CoordinationModel.load(path)
    pose_meta=Path(path).parent/'config.json'
    if pose_meta.exists() and json.loads(pose_meta.read_text()).get('policy_variant')=='pose-bottleneck-driving-v1':
        from .driving_pose_policy import PoseModel
        return PoseModel.load(path)
    metadata=Path(path).parent/'config.json'
    if metadata.exists() and json.loads(metadata.read_text()).get('policy_variant') in ('phase-memory-driving-v1','phase-memory-local-hand-v2'):
        from .driving_memory import MemoryModel
        return MemoryModel.load(path)
    if metadata.exists() and json.loads(metadata.read_text()).get('policy_variant')=='recurrent-driving-v1':
        from .driving_recurrent import RecurrentModel
        return RecurrentModel.load(path)
    if metadata.exists() and json.loads(metadata.read_text()).get('policy_variant')=='normalized-driving-v1':
        from .driving_policy import DrivingTeacherPolicy
        policy_class=DrivingTeacherPolicy
    return PPO.load(path, env=env, device="cpu", custom_objects={"policy_class": policy_class})


def _initialize_workers(policy, config, env_class, reset_options):
    global _worker_policy, _worker_config, _worker_env_class, _worker_reset_options
    torch.set_num_threads(1)
    _worker_policy, _worker_config = policy, config
    _worker_env_class, _worker_reset_options = env_class, reset_options


def _evaluate_job(job):
    support, seed, case, seconds, amplitude = job
    return run_case(_worker_policy, _worker_config, case, seed, seconds, amplitude, support,
                    fixed=_worker_policy is None, env_class=_worker_env_class, reset_options=_worker_reset_options)


def _evaluations(model, config, jobs, workers, env_class, reset_options):
    if workers == 1:
        for support, seed, case, seconds, amplitude in jobs:
            yield run_case(model,config,case,seed,seconds,amplitude,support,fixed=model is None,
                           env_class=env_class,reset_options=reset_options)
    else:
        # PPO.predict delegates directly to policy.predict. Passing the policy
        # avoids copying training environments or optimizer handles to workers.
        policy = model.policy if model is not None else None
        with multiprocessing.get_context("fork").Pool(
                workers, initializer=_initialize_workers, initargs=(policy,config,env_class,reset_options)) as pool:
            yield from pool.imap(_evaluate_job,jobs)


def run_case(model, config, case, seed, seconds, amplitude=10., support="both", fixed=False,
             env_class=SteeringTeacherEnv, reset_options=None, trace_fields=()):
    from copy import deepcopy
    config = deepcopy(config)
    config.episode_seconds = seconds
    config.amplitude_deg = amplitude
    config.target_mode = "hold" if case == "hold" else "sequence"
    config.shaping_weight = 0.
    env = env_class(config)
    free = {"both": -1, "left": 1, "right": 0}[support]
    options = {"direction": -1 if case in ("negative", "sequence_negative") else 1,
               "exact_curriculum": True, "free_side": free}
    if case in ("positive", "negative"):
        options["mode"] = "step"
    options.update(reset_options or {})
    rows = []
    try:
        observation, preparation = env.reset(seed=seed, options=options)
        if hasattr(model,"reset"):model.reset()
        initial_angle = env.wheel_angle
        if fixed:
            env.controller.update = lambda data, action, dt: None
        done = False
        for _ in range(round(seconds*config.policy_hz)):
            action = np.zeros(24) if fixed else model.predict(observation, deterministic=True)[0]
            observation, _, done, truncated, info = env.step(action)
            row = {key: info[key] for key in (
                "time", "wheel_angle", "target_angle", "error_deg", "left_force", "right_force",
                "physics_peak_error_deg", "physics_min_arm_margin_deg", "physics_collision_peak_n",
                "termination_reason", "qp_failures", "qp_position_error_mm", "collision_qp_failures",
                "collision_constraints", "workspace_clips")}
            row["free_force"] = info.get("physics_free_hand_force_peak_n", info["right_force"] if free==1 else info["left_force"])
            row.update({key:info[key] for key in trace_fields})
            rows.append(row)
            if done or truncated:
                break
        errors = np.array([row["error_deg"] for row in rows])
        times = np.array([row["time"] for row in rows])
        complete = bool(not done and times[-1]+1e-8 >= seconds)
        tail = times >= max(0., seconds-3)-1e-8
        peak = max(row["physics_peak_error_deg"] for row in rows)
        tail_peak = max((row["physics_peak_error_deg"] for row, use in zip(rows,tail) if use), default=None) if complete else None
        rmse = float(np.sqrt(np.mean(errors**2)))
        free_force = max(row["free_force"] for row in rows) if free >= 0 else None
        progress = float((env.wheel_angle-initial_angle)*options["direction"]/np.deg2rad(amplitude)) if case in ("positive", "negative") else None
        passed = complete and ((peak <= 1.) if case == "hold" else (rmse <= 1. and tail_peak <= 1.))
        if progress is not None:
            passed = passed and progress >= .5
        if free >= 0:
            passed = passed and free_force <= .2
        record = dict(case=case, seed=seed, seconds=seconds, amplitude_deg=amplitude, support=support,
                      complete=complete, rmse_deg=rmse, peak_error_deg=peak, tail_peak_deg=tail_peak,
                      min_arm_margin_deg=min(row["physics_min_arm_margin_deg"] for row in rows),
                      collision_peak_n=max(row["physics_collision_peak_n"] for row in rows),
                      free_hand_peak_force_n=free_force, signed_progress_fraction=progress,
                      termination_reason=info["termination_reason"], passed=bool(passed),
                      qp_failures=info["qp_failures"], collision_qp_failures=info["collision_qp_failures"],
                      collision_constraints=info["collision_constraints"], workspace_clips=info["workspace_clips"],
                      peak_qp_position_error_mm=max(r["qp_position_error_mm"] for r in rows),
                      reset_options=reset_options or {},
                      preparation={key:preparation[key] for key in ("wheel_angle", "left_force", "right_force")})
        return record, rows
    finally:
        env.close()


def evaluation_cases(include_reverse_sequence=False):
    cases=[("hold",3.,5.),("hold",10.,5.),("positive",8.,5.),("negative",8.,5.),
           ("positive",10.,10.),("negative",10.,10.),("sequence",22.,10.)]
    if include_reverse_sequence:
        cases.append(("sequence_negative",22.,10.))
    return cases


def suite(model, config, seeds=(1007,), supports=("both",), output=None, baselines=None,
          independent=False, env_class=SteeringTeacherEnv, reset_options=None, workers=1,
          include_reverse_sequence=False):
    cases=evaluation_cases(include_reverse_sequence)
    records=[]
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
    jobs=[(support,seed,case,seconds,amplitude) for support in supports for seed in seeds
          for case,seconds,amplitude in cases]
    for job,(record,rows) in zip(jobs,_evaluations(model,config,jobs,workers,env_class,reset_options)):
        support,seed,case,seconds,amplitude=job
        records.append(record)
        if output:
            (output/"progress.json").write_text(json.dumps(records,indent=2))
            # Compact control traces; no 500Hz dumps or duplicate video.
            np.savez_compressed(output/f"{support}_{case}_{amplitude:g}_{seconds:g}_s{seed}.npz",
                                time=[r["time"] for r in rows], angle=[r["wheel_angle"] for r in rows],
                                target=[r["target_angle"] for r in rows])
        print(json.dumps(record),flush=True)
    improvements={}
    for support in supports:
        for direction in ("positive","negative"):
            selected=[r for r in records if r["case"]==direction and r["support"]==support]
            if baselines:
                paired=[b["rmse_deg"] for r in selected for b in baselines["cases"]
                        if all(b[key]==r[key] for key in ("case","support","seed","seconds","amplitude_deg"))]
                if len(paired)!=len(selected):
                    raise ValueError("Fixed-target baseline must match every evaluation case")
                improvements[f"{support}/{direction}"]=float(1-np.mean([r["rmse_deg"] for r in selected])/max(np.mean(paired),1e-9))
    passed=all(r["passed"] for r in records) and (not baselines or all(v>=.3 for v in improvements.values()))
    summary=dict(cases=records,passed=bool(passed),independent_acceptance=independent,
                 baseline_improvements=improvements,task=asdict(config),
                 protocol="r2-v2-both-sequence-signs" if include_reverse_sequence else "r1-r2-v1")
    if output:
        (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--stiffness",type=float,default=.3)
    parser.add_argument("--legacy-limits",action="store_true")
    parser.add_argument("--seeds",nargs="+",type=int,default=[1007])
    parser.add_argument("--supports",nargs="+",choices=["both","left","right"],default=["both"])
    parser.add_argument("--hand-mode",choices=["absolute","leaky"])
    parser.add_argument("--hand-alpha",type=float)
    parser.add_argument("--include-reverse-sequence",action="store_true")
    parser.add_argument("--hand-lead-rad",type=float)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.hand_mode is None and args.model and (args.model.parent/"config.json").exists():
        task=json.loads((args.model.parent/"config.json").read_text()).get("task",{})
        args.hand_mode=task.get("hand_action_mode","absolute")
        if args.hand_alpha is None:
            args.hand_alpha=task.get("hand_leak_alpha",.95)
        if args.hand_lead_rad is None:
            args.hand_lead_rad=task.get("hand_joint_lead_rad")
    suite(load_teacher(args.model) if args.model else None,
          TaskConfig(centering_stiffness=args.stiffness,strict_safety=not args.legacy_limits,
                     hand_action_mode=args.hand_mode or "absolute",hand_leak_alpha=args.hand_alpha or .95,
                     hand_joint_lead_rad=args.hand_lead_rad),
          seeds=args.seeds,supports=args.supports,output=args.output,
          include_reverse_sequence=args.include_reverse_sequence)
