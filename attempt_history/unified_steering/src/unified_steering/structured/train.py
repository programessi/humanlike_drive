"""Bounded R1/R2 teacher training with held-out gates and TensorBoard."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import shutil
import importlib.metadata

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import SteeringTeacherEnv, TaskConfig
from .evaluate import load_teacher, suite
from .parking import SupportTeacherEnv, SCREENED_RESETS
from .curriculum import balanced_condition


def validate_mode_probabilities(values, hold_fraction=None):
    if values is None:
        return None
    probabilities = np.asarray(values, dtype=float)
    if (probabilities.shape != (4,) or not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1., atol=1e-9, rtol=0)):
        raise ValueError("Specify four positive probabilities summing to one: hold step sine sequence")
    if hold_fraction is not None:
        raise ValueError("Use either mode probabilities or hold fraction, not both")
    return tuple(float(value) for value in probabilities)


class CourseEnv(SupportTeacherEnv):
    def __init__(self, stage, parking="nominal", hold_fraction=None, angle_sigma=4., hand_mode="absolute",hand_alpha=.95,mode_probabilities=None,hand_lead=None):
        mode_probabilities = validate_mode_probabilities(mode_probabilities, hold_fraction)
        super().__init__(TaskConfig(amplitude_deg=10., episode_seconds=22., randomization_strength=0.,
                                   shaping_weight=0.,angle_reward_sigma_deg=angle_sigma,hand_action_mode=hand_mode,
                                   hand_leak_alpha=hand_alpha,hand_joint_lead_rad=hand_lead))
        self.stage=stage
        self.parking=parking
        self.hold_fraction=hold_fraction
        self.mode_probabilities=mode_probabilities

    def reset(self, seed=None, options=None):
        # Single-hand labels determine only the reset distribution. No role or
        # stage label is supplied to the actor and no hand is locked at runtime.
        if seed is not None or not hasattr(self,"np_random"):
            rng=np.random.default_rng(seed)
        else:
            rng=self.np_random
        chosen=dict(options or {})
        # Preserve the already feasible five-degree motions during adaptation.
        self.config.amplitude_deg=float(chosen.pop("training_amplitude_deg",rng.choice([5.,10.])))
        if self.mode_probabilities is not None:
            chosen.setdefault("mode",str(rng.choice(["hold","step","sine","sequence"],
                                                    p=self.mode_probabilities)))
        elif self.hold_fraction is not None:
            remaining=1-self.hold_fraction
            chosen.setdefault("mode",str(rng.choice(["hold","step","sine","sequence"],
                              p=[self.hold_fraction,remaining*.4/.75,remaining*.1/.75,remaining*.25/.75])))
        if self.stage=="R2" and "free_side" not in chosen:
            chosen["free_side"]=int(rng.choice([-1,0,1],p=[.25,.375,.375]))
        if self.stage=="R2" and self.parking=="screened" and chosen.get("free_side",-1)>=0:
            condition=str(rng.choice(["nominal",*SCREENED_RESETS]))
            for key,value in SCREENED_RESETS.get(condition,{}).items():
                chosen.setdefault(key,value)
        return super().reset(seed=seed, options=chosen)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(course_mode=self.episode_mode, course_amplitude_deg=self.config.amplitude_deg,
                    course_direction=self.episode_direction)
        if "course_block" in self.reset_options:
            info.update(course_block=self.reset_options["course_block"],course_slot=self.reset_options["course_slot"])
        return observation, reward, terminated, truncated, info


class BalancedCourseEnv(CourseEnv):
    def __init__(self,*args,worker_index=0,workers=4,curriculum_seed=7,**kwargs):
        super().__init__(*args,**kwargs)
        self.worker_index,self.workers,self.curriculum_seed=worker_index,workers,curriculum_seed
        self.course_episode_index=0

    def reset(self,seed=None,options=None):
        if seed is not None:
            self.course_episode_index=0
        index=self.worker_index+self.workers*self.course_episode_index
        selected=balanced_condition(index,self.curriculum_seed)
        selected.update(options or {})
        observation,info=super().reset(seed=seed,options=selected)
        self.course_episode_index+=1
        return observation,info


def factory(stage,seed,parking="nominal",hold_fraction=None,angle_sigma=4.,hand_mode="absolute",hand_alpha=.95,mode_probabilities=None,
            balanced=False,worker_index=0,workers=4,curriculum_seed=7,hand_lead=None):
    def make():
        cls=BalancedCourseEnv if balanced else CourseEnv
        kwargs=dict(worker_index=worker_index,workers=workers,curriculum_seed=curriculum_seed) if balanced else {}
        env=cls(stage,parking,hold_fraction,angle_sigma,hand_mode,hand_alpha,mode_probabilities,hand_lead,**kwargs)
        env.reset(seed=seed)
        return Monitor(env,info_keywords=("termination_reason",))
    return make


def write_json(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def stage_suite(model,stage,parking,seeds,output,baselines=None,independent=False,hand_mode="absolute",hand_alpha=.95,angle_sigma=4.,include_reverse_sequence=False,hand_lead=None):
    """Nominal gates plus explicitly named expanded-reset conditions for R2."""
    plans=[("nominal",{},("both",) if stage=="R1" else ("both","left","right"))]
    if stage=="R2" and parking=="screened":
        plans += [(name,options,("left","right")) for name,options in SCREENED_RESETS.items()]
    output.mkdir(parents=True,exist_ok=False)
    result=dict(cases=[],conditions={},baseline_improvements={},independent_acceptance=independent,passed=True)
    for name,options,supports in plans:
        part=suite(model,TaskConfig(hand_action_mode=hand_mode,hand_leak_alpha=hand_alpha,angle_reward_sigma_deg=angle_sigma,hand_joint_lead_rad=hand_lead),seeds=seeds,supports=supports,output=output/name,
                   baselines=baselines["conditions"][name] if baselines else None,
                   independent=independent,env_class=SupportTeacherEnv,reset_options=options,workers=4,
                   include_reverse_sequence=include_reverse_sequence)
        result["conditions"][name]=part
        result["cases"] += part["cases"]
        result["baseline_improvements"].update({name+"/"+k:v for k,v in part["baseline_improvements"].items()})
        result["passed"]=result["passed"] and part["passed"]
    write_json(output/"summary.json",result)
    return result


class GateCallback(BaseCallback):
    def __init__(self,output,stage,baseline,every=16384,parking="nominal",hand_mode="absolute",hand_alpha=.95,angle_sigma=4.,development_seeds=(1007,),qualification_rmse=1.,include_reverse_sequence=False,hand_lead=None):
        super().__init__()
        self.output,self.stage,self.baseline,self.every=output,stage,baseline,every
        self.parking=parking
        self.hand_mode=hand_mode
        self.hand_alpha=hand_alpha
        self.angle_sigma=angle_sigma
        self.development_seeds=tuple(development_seeds)
        self.qualification_rmse=qualification_rmse
        self.include_reverse_sequence=include_reverse_sequence
        self.hand_lead=hand_lead
        self.last=-1
        self.consecutive=0
        self.qualified=False
        self.best_score=np.inf
        self.best_response_score=np.inf
        self.records=[]
        self.stale_bad=0
        self.stop_reason=""
        self.started=time.monotonic()

    def _on_rollout_start(self):
        if self.num_timesteps==self.last or self.num_timesteps%self.every:
            return
        self.check()

    def check(self,final=False):
        if self.num_timesteps==self.last:
            return
        self.last=self.num_timesteps
        torch_state=torch.get_rng_state();numpy_state=np.random.get_state();python_state=random.getstate()
        supports=("both",) if self.stage=="R1" else ("both","left","right")
        try:
            result=stage_suite(self.model,self.stage,self.parking,self.development_seeds,
                         self.output/"development"/f"step_{self.num_timesteps:09d}",baselines=self.baseline,
                         hand_mode=self.hand_mode,hand_alpha=self.hand_alpha,angle_sigma=self.angle_sigma,
                         include_reverse_sequence=self.include_reverse_sequence,hand_lead=self.hand_lead)
        finally:
            torch.set_rng_state(torch_state);np.random.set_state(numpy_state);random.setstate(python_state)
        records=result["cases"]
        holds=[r for r in records if r["case"]=="hold"]
        safe=all(r["complete"] for r in records) and all(r["peak_error_deg"]<=1 for r in holds)
        retained_holds=[r for r in holds if self.stage=="R1" or r["support"]=="both"]
        retention_safe=all(r["peak_error_deg"]<=1 and r["complete"] for r in retained_holds)
        score=max(np.mean([r["rmse_deg"] for r in records if r["case"]==direction]) for direction in ("positive","negative","sequence"))
        directions=("positive","negative","sequence")
        if self.include_reverse_sequence:
            directions+= ("sequence_negative",)
            safe=safe and all(r.get("free_hand_peak_force_n") is None or r["free_hand_peak_force_n"]<=.2 for r in records)
            group_keys={(r["support"],r["case"]) for r in records if r["case"]!="hold"}
            score=max(float(np.mean([r["rmse_deg"] for r in records if (r["support"],r["case"])==key])) for key in group_keys)
            if any(not r["complete"] for r in records):
                # Explicit failure-priority selection score, not an RMSE.
                score+=1000.
        improved=safe and score<self.best_score
        response_improved=bool(score<self.best_response_score-1e-6)
        self.best_response_score=min(self.best_response_score,score)
        if improved:
            self.best_score=score
            self.model.save(self.output/"best")
        qualification_passed=result["passed"] and all(r["case"]=="hold" or r["rmse_deg"]<=self.qualification_rmse for r in records)
        self.consecutive=self.consecutive+1 if qualification_passed else 0
        if result["passed"]:
            self.model.save(self.output/"passing_development")
        self.qualified=self.consecutive>=2
        # Failure to acquire a NEW single-support skill is not forgetting R1.
        # Stop for retained-hold regression or physical/solver failures only
        # when command response has also stopped improving.
        failures=sum(not r["complete"] for r in records)
        self.stale_bad=self.stale_bad+1 if (not retention_safe or failures) and not response_improved else 0
        if self.qualified:
            self.model.save(self.output/"qualified_development")
        if self.stale_bad>=3:
            self.stop_reason="three_bad_evaluations_requires_diagnosis"
        for direction in directions:
            self.logger.record(f"development/{direction}/rmse_deg",float(np.mean([r["rmse_deg"] for r in records if r["case"]==direction])))
        self.logger.record("development/hold_peak_deg",max(r["peak_error_deg"] for r in holds))
        self.logger.record("development/passed",float(result["passed"]))
        self.logger.record("development/qualification_passed",float(qualification_passed))
        self.logger.record("development/hard_failures",sum(not r["complete"] for r in records))
        self.logger.dump(self.num_timesteps)
        record=dict(steps=self.num_timesteps,passed=result["passed"],qualified=self.qualified,
                    qualification_passed=bool(qualification_passed),
                    hold_safe=safe,score=float(score),best_score=float(self.best_score) if np.isfinite(self.best_score) else None,stop_reason=self.stop_reason,
                    retention_hold_safe=retention_safe,command_response_improved=response_improved,
                    hard_failures=failures,
                    elapsed_seconds=time.monotonic()-self.started)
        self.records.append(record);write_json(self.output/"development_index.json",self.records)
        self.model.save(self.output/"latest")
        print('GATE '+json.dumps(record),flush=True)

    def _on_step(self):
        for env_index,info in enumerate(self.locals["infos"]):
            for name in ("error_deg","physics_min_arm_margin_deg","physics_collision_peak_n"):
                self.logger.record_mean(f"physical/{name}",info[name])
            for name,value in info.items():
                if name.startswith("reward_"):
                    self.logger.record_mean(f"reward_raw/{name[7:]}",value)
                    self.logger.record_mean(f"reward_scaled/{name[7:]}",value*.05)
            for name in ("qp_failures","collision_qp_failures","workspace_clips"):
                self.logger.record_mean(f"controller/{name}",info[name])
            if "episode" in info:
                with (self.output/"episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps({"steps":self.num_timesteps,"environment":env_index,
                                  "free_side":info["preparation_free_side"],
                                  "phase_deg":info["preparation_phase_deg"],
                                  "mode":info.get("course_mode"),
                                  "amplitude_deg":info.get("course_amplitude_deg"),
                                  "direction":info.get("course_direction"),
                                  "course_block":info.get("course_block"),"course_slot":info.get("course_slot"),
                                  "tail_error_deg":info["error_deg"],**info["episode"]})+'\n')
        return not self.qualified and not self.stop_reason


def run(args):
    torch.set_num_threads(1)
    mode_probabilities=validate_mode_probabilities(args.mode_probabilities,args.hold_fraction)
    args.output.mkdir(parents=True,exist_ok=False)
    task=TaskConfig(episode_seconds=22.,angle_reward_sigma_deg=args.angle_sigma,hand_action_mode=args.hand_mode,hand_leak_alpha=args.hand_alpha,hand_joint_lead_rad=args.hand_lead_rad)
    metadata=dict(stage=args.stage,seed=args.seed,requested_steps=args.steps,envs=args.envs,
                  training_worker_seeds=[args.seed+i for i in range(args.envs)],
                  parking=args.parking,hold_fraction=args.hold_fraction,mode_probabilities=mode_probabilities,
                  balanced_course=args.balanced_course,
                  effective_runtime=dict(episode_seconds=22.,strict_joint_lead_rad=float(np.deg2rad(5.)),policy_hz=20,robot_hz=50,physics_hz=500),
                  parent=str(args.resume.resolve()),parent_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                  task=asdict(task),status="preflight",r1_passed=False,r2_passed=False,
                  interface="structured-wrist-teacher-v4-separate-hand-lead" if args.hand_lead_rad is not None else "structured-wrist-teacher-v3-incremental-hand" if args.hand_mode=="leaky" else "structured-wrist-teacher-v2-relative-strict",learning_rate=args.lr,
                  reset_hand_head=args.reset_hand_head,
                  wrist_std_override=args.wrist_std,
                  evaluation=dict(development_seeds=args.development_seeds,qualification_rmse=args.qualification_rmse,
                                  independent_seeds=args.acceptance_seeds,every_steps=16384,
                                  include_reverse_sequence=args.include_reverse_sequence,
                                  selection="stable complete cases, then worst support/case mean" if args.include_reverse_sequence else "legacy v1"),
                  note="Teacher is privileged; no deployment/student success is claimed.")
    write_json(args.output/"config.json",metadata)
    sources={}
    root=Path(__file__).resolve().parents[1]
    project=root.parents[1]
    for name in ("r1_r2_v1.json","r2_screened_v1.json","r2_leaky_hand_v1.json","r2_repair_v2.json","r22_balanced_course_v2.json","r22_separate_hand_lead_v2.json"):
        source=project/"configs"/name
        if source.exists():
            destination=args.output/"frozen_configs"/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    for source in list((root/"structured").rglob("*.py"))+[root/"wheel_load.py"]:
        relative=source.relative_to(root)
        destination=args.output/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        sources[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(args.output/"provenance.json",dict(source_sha256=sources,
               dependencies={name:importlib.metadata.version(name) for name in ("mujoco","numpy","scipy","torch","gymnasium","stable-baselines3")},
               tensorboard="http://localhost:6006"))
    model=load_teacher(args.resume)
    parent_config=args.resume.parent/"config.json"
    parent_mode=json.loads(parent_config.read_text()).get("task",{}).get("hand_action_mode","absolute") if parent_config.exists() else "absolute"
    if parent_mode!=args.hand_mode and not args.reset_hand_head:
        raise ValueError("Changed hand semantics require explicit --reset-hand-head transfer initialization")
    if args.reset_hand_head:
        with torch.no_grad():
            model.policy.action_net.weight[12:]=0.
            model.policy.action_net.bias[12:]=0.
            for parameter in (model.policy.action_net.weight,model.policy.action_net.bias):
                for value in model.policy.optimizer.state[parameter].values():
                    if isinstance(value,torch.Tensor) and value.shape==parameter.shape:
                        value[12:]=0.
    if args.wrist_std is not None:
        with torch.no_grad():
            model.policy.log_std[:12]=float(np.log(args.wrist_std))
            for value in model.policy.optimizer.state[model.policy.log_std].values():
                if isinstance(value,torch.Tensor) and value.shape==model.policy.log_std.shape:
                    value[:12]=0.
    model.save(args.output/"initial")
    supports=("both",) if args.stage=="R1" else ("both","left","right")
    baseline=stage_suite(None,args.stage,args.parking,tuple(args.development_seeds),args.output/"fixed_targets",hand_mode=args.hand_mode,hand_alpha=args.hand_alpha,angle_sigma=args.angle_sigma,include_reverse_sequence=args.include_reverse_sequence,hand_lead=args.hand_lead_rad)
    vec=SubprocVecEnv([factory(args.stage,args.seed+1000*i,args.parking,args.hold_fraction,args.angle_sigma,args.hand_mode,args.hand_alpha,mode_probabilities,
                             args.balanced_course,i,args.envs,args.seed,args.hand_lead_rad) for i in range(args.envs)],start_method="fork")
    try:
        model.set_env(vec)
        model.n_steps=256;model.batch_size=256;model.n_epochs=4
        model.learning_rate=args.lr;model.lr_schedule=lambda _:args.lr
        for group in model.policy.optimizer.param_groups:
            group['lr']=args.lr
        model.tensorboard_log=str(args.output/"tensorboard")
        model.set_random_seed(args.seed)
        callback=GateCallback(args.output,args.stage,baseline,parking=args.parking,hand_mode=args.hand_mode,hand_alpha=args.hand_alpha,angle_sigma=args.angle_sigma,
                              development_seeds=args.development_seeds,qualification_rmse=args.qualification_rmse,
                              include_reverse_sequence=args.include_reverse_sequence,hand_lead=args.hand_lead_rad)
        metadata.update(status="training",started_unix=time.time())
        write_json(args.output/"config.json",metadata)
        model.learn(args.steps,callback=callback,reset_num_timesteps=True,tb_log_name=args.stage)
        if not callback.qualified and not callback.stop_reason:
            callback.check(final=True)
        model.save(args.output/"final")
        metadata.update(status="development_qualified" if callback.qualified else "training_requires_diagnosis",
                        actual_steps=model.num_timesteps,stop_reason=callback.stop_reason,ended_unix=time.time())
        if callback.qualified:
            qualified=load_teacher(args.output/"qualified_development.zip")
            acceptance_baseline=stage_suite(None,args.stage,args.parking,tuple(args.acceptance_seeds),
                                      args.output/"acceptance_fixed_targets",independent=True,hand_mode=args.hand_mode,hand_alpha=args.hand_alpha,angle_sigma=args.angle_sigma,include_reverse_sequence=args.include_reverse_sequence,hand_lead=args.hand_lead_rad)
            acceptance=stage_suite(qualified,args.stage,args.parking,tuple(args.acceptance_seeds),
                             args.output/"acceptance",baselines=acceptance_baseline,independent=True,hand_mode=args.hand_mode,hand_alpha=args.hand_alpha,angle_sigma=args.angle_sigma,include_reverse_sequence=args.include_reverse_sequence,hand_lead=args.hand_lead_rad)
            metadata[args.stage.lower()+"_passed"]=acceptance["passed"] and (args.stage=="R1" or args.parking=="screened")
            metadata['status']="stage_passed" if acceptance["passed"] else "acceptance_failed"
            if args.stage=="R2" and args.parking=="nominal" and acceptance["passed"]:
                metadata.update(status="nominal_single_support_passed_requires_reset_expansion",nominal_single_support_passed=True)
    except BaseException as error:
        metadata.update(status="failed",error=repr(error))
        raise
    finally:
        vec.close()
        write_json(args.output/"config.json",metadata)
    print(json.dumps(metadata),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["R1","R2"],default="R1")
    parser.add_argument("--resume",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--steps",type=int,default=131072)
    parser.add_argument("--envs",type=int,default=4)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--parking",choices=["nominal","screened"],default="nominal")
    parser.add_argument("--hold-fraction",type=float)
    parser.add_argument("--mode-probabilities",type=float,nargs=4,metavar=("HOLD","STEP","SINE","SEQUENCE"))
    parser.add_argument("--angle-sigma",type=float,default=4.)
    parser.add_argument("--hand-mode",choices=["absolute","leaky"],default="absolute")
    parser.add_argument("--reset-hand-head",action="store_true")
    parser.add_argument("--hand-alpha",type=float,default=.95)
    parser.add_argument("--acceptance-seeds",type=int,nargs=3,default=[1107,1117,1127])
    parser.add_argument("--development-seeds",type=int,nargs="+",default=[1007])
    parser.add_argument("--qualification-rmse",type=float,default=1.)
    parser.add_argument("--wrist-std",type=float)
    parser.add_argument("--include-reverse-sequence",action="store_true")
    parser.add_argument("--balanced-course",action="store_true")
    parser.add_argument("--hand-lead-rad",type=float)
    args=parser.parse_args()
    if not 1<=args.envs<=8 or args.steps<=0 or not 0<args.lr<=3e-4:
        parser.error("Use bounded positive budgets and learning rates")
    if args.hold_fraction is not None and not 0<args.hold_fraction<1:
        parser.error("Holding mixture must lie strictly between zero and one")
    if args.balanced_course and (args.stage!="R2" or args.hold_fraction is not None or args.mode_probabilities is not None):
        parser.error("Balanced R2 course has a frozen mixture; omit hold/mode overrides")
    if args.hand_lead_rad is not None and (not np.isfinite(args.hand_lead_rad) or not 0<args.hand_lead_rad<=.25):
        parser.error("Hand target lead must lie within (0,.25] rad")
    try:
        validate_mode_probabilities(args.mode_probabilities,args.hold_fraction)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.angle_sigma) or not 0<args.angle_sigma<=10:
        parser.error("Angle reward width must be within (0,10] degrees")
    if not 0<args.hand_alpha<=1:
        parser.error("Hand leakage coefficient must lie in (0,1]")
    if len(set(args.acceptance_seeds))!=3 or set(args.development_seeds)&set(args.acceptance_seeds):
        parser.error("Acceptance needs three distinct held-out seeds excluding development seeds")
    if not 0<args.qualification_rmse<=1 or (args.wrist_std is not None and not 0<args.wrist_std<=.5):
        parser.error("Qualification RMSE must be within (0,1]; wrist std override within (0,0.5]")
    run(args)
