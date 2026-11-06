"""Versioned, bounded R2 repair experiments; no held-out tuning."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import multiprocessing

import numpy as np
import torch

from .env import TaskConfig
from .evaluate import load_teacher, suite, run_case
from .interventions import DiagnosticConfig, DiagnosticEnv, INTERVENTIONS, TRACE_FIELDS
from .parking import SupportTeacherEnv


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def groups(summary):
    result=[]
    for support in ("both","left","right"):
        for case in ("hold","positive","negative","sequence","sequence_negative"):
            rows=[r for r in summary["cases"] if r["support"]==support and r["case"]==case]
            if not rows:
                continue
            completed=[r for r in rows if r["complete"]]
            result.append(dict(support=support,case=case,count=len(rows),
                               complete=len(completed),precision_passes=sum(r["passed"] for r in rows),
                               complete_mean_rmse_deg=float(np.mean([r["rmse_deg"] for r in completed])) if completed else None,
                               complete_max_rmse_deg=max((r["rmse_deg"] for r in completed),default=None),
                               complete_tail_peak_deg=max((r["tail_peak_deg"] for r in completed),default=None),
                               reasons=[r["termination_reason"] for r in rows if not r["complete"]]))
    return result


def compare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text())
    shutil.copy2(args.protocol,args.output/"protocol.json")
    task=TaskConfig(episode_seconds=22.,hand_action_mode="leaky",hand_leak_alpha=.995,angle_reward_sigma_deg=2.)
    seeds=tuple(protocol["development_seeds"])
    baseline=suite(None,task,seeds=seeds,supports=tuple(protocol["supports"]),
                   output=args.output/"fixed_targets",env_class=SupportTeacherEnv,
                   workers=args.workers,include_reverse_sequence=True)
    results={}
    for name,source in protocol["models"].items():
        model_path=Path(source)
        summary=suite(load_teacher(model_path),task,seeds=seeds,supports=tuple(protocol["supports"]),
                      output=args.output/name,env_class=SupportTeacherEnv,baselines=baseline,
                      workers=args.workers,include_reverse_sequence=True)
        results[name]=dict(model=str(model_path),sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                           cases=len(summary["cases"]),complete=sum(r["complete"] for r in summary["cases"]),
                           precision_passes=sum(r["passed"] for r in summary["cases"]),
                           groups=groups(summary))
        write_json(args.output/"comparison.json",results)
        print("COMPARISON "+json.dumps({name:results[name]}),flush=True)
    lines=["# R2.0 检查点回测（v2开发协议）","",
           "冻结模型；仅使用已查看的开发种子。每模型72例，包含正、反起始连续序列，不是独立验收。", "",
           "| 模型 | 完整执行 | 1°精度达标 |", "| --- | --- | --- |"]
    for name,result in results.items():
        lines.append(f"| {name} | {result['complete']}/72 | {result['precision_passes']}/72 |")
    lines += ["", "未完成任务的前缀RMSE不与完整任务排名。细分见comparison.json，各模型目录包含原始summary和20Hz轨迹。"]
    (args.output/"README.md").write_text("\n".join(lines)+"\n")


def _intervention_job(job):
    model_path,intervention,seed,start=job
    torch.set_num_threads(1)
    task=DiagnosticConfig(episode_seconds=22.,hand_action_mode="leaky",hand_leak_alpha=.995,
                          angle_reward_sigma_deg=2.,intervention=intervention,
                          intervention_start_seconds=start)
    record,trace=run_case(load_teacher(model_path),task,"sequence_negative",seed,22.,10.,"right",
                          env_class=DiagnosticEnv,trace_fields=TRACE_FIELDS)
    record.update(intervention=intervention,intervention_start_seconds=start,
                  executed_seconds=trace[-1]["time"],
                  max_radius_m=max(row["audit_radius_m"][1] for row in trace),
                  radius_end_m=trace[-1]["audit_radius_m"][1],
                  clips_by_side=trace[-1]["audit_workspace_clips"],
                  hand_lead_hits_by_side=trace[-1]["audit_hand_lead_hits"])
    return record,trace


def intervene(args):
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text())
    model_path=args.model or Path(protocol["models"]["sequence_final"])
    shutil.copy2(args.protocol,args.output/"protocol.json")
    kinds=tuple(protocol.get("diagnostic_interventions",INTERVENTIONS))
    if not kinds or any(kind not in INTERVENTIONS for kind in kinds):
        raise ValueError("Unknown diagnostic intervention list")
    jobs=[(model_path,kind,seed,args.start_seconds) for kind in kinds
          for seed in protocol["development_seeds"]]
    results=[]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for record,trace in pool.imap(_intervention_job,jobs):
            results.append(record)
            name=f"{record['intervention']}_s{record['seed']}"
            np.savez_compressed(args.output/(name+".npz"),**{
                key:[row[key] for row in trace] for key in trace[0] if key!="termination_reason"})
            write_json(args.output/"summary.json",dict(model=str(model_path),
                       model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                       diagnostic_only=True,cases=results))
            print("INTERVENTION "+json.dumps(record),flush=True)
    lines=["# R2.1 冻结策略干预","",
           "诊断分支有明确动作/目标覆盖，不能作为统一学习策略通过证据。相同模型、初态与参考；实际执行仍有限速和领先约束。", "",
           "| 干预 | 完整执行 | 精度达标 | 最早结束(s) | 最大支撑腕径向距离(m) |",
           "| --- | --- | --- | --- | --- |"]
    for kind in kinds:
        rows=[r for r in results if r["intervention"]==kind]
        lines.append(f"| {kind} | {sum(r['complete'] for r in rows)}/3 | {sum(r['passed'] for r in rows)}/3 | {min(r['executed_seconds'] for r in rows):.2f} | {max(r['max_radius_m'] for r in rows):.4f} |")
    (args.output/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["compare","intervene"])
    parser.add_argument("--protocol",type=Path,default=Path("configs/r2_repair_v2.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--start-seconds",type=float,default=0.)
    args=parser.parse_args()
    if not 1<=args.workers<=8:
        parser.error("Use 1 to 8 workers")
    torch.set_num_threads(1)
    if args.start_seconds<0:
        parser.error("Intervention start must be nonnegative")
    (compare if args.command=="compare" else intervene)(args)
