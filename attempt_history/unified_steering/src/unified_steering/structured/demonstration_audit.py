"""Evaluate the explicit scripted demonstrator through the standard task gates."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

import torch

from .bootstrap import task_config
from .demonstrator import Demonstrator, DemonstratorConfig
from .evaluate import evaluation_cases, run_case
from .parking import SCREENED_RESETS, SupportTeacherEnv


def job(arguments):
    protocol, condition, options, support, seed, case, seconds, amplitude = arguments
    torch.set_num_threads(1)
    holder = {}
    class Environment(SupportTeacherEnv):
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            holder["expert"] = Demonstrator(self, DemonstratorConfig(**protocol["demonstrator"]))
            return result
    class Proxy:
        def predict(self, observation, deterministic=True):
            return holder["expert"].predict(observation, deterministic)
    record,_=run_case(Proxy(),task_config(protocol),case,seed,seconds,amplitude,support,
                       env_class=Environment,reset_options=options)
    record["condition"]=condition
    return record


def run(args):
    protocol=json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True,exist_ok=args.resume)
    frozen=args.output/"protocol.json"
    if frozen.exists() and json.loads(frozen.read_text())!=protocol:
        raise ValueError("Cannot resume with a changed protocol")
    shutil.copy2(args.protocol,frozen)
    plans=[("nominal",{},("both","left","right"))]
    plans += [(name,options,("left","right")) for name,options in SCREENED_RESETS.items()]
    jobs=[(protocol,name,options,support,seed,case,seconds,amplitude)
        for name,options,supports in plans for support in supports
        for seed in protocol["development_seeds"] for case,seconds,amplitude in evaluation_cases(True)]
    rows=json.loads((args.output/"summary.json").read_text())["cases"] if args.resume and (args.output/"summary.json").exists() else []
    def key(row):return (row["condition"],row["support"],row["seed"],row["case"],row["seconds"],row["amplitude_deg"])
    completed={key(row) for row in rows}
    jobs=[entry for entry in jobs if (entry[1],entry[3],entry[4],entry[5],entry[6],entry[7]) not in completed]
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(job,jobs):
            rows.append(record)
            (args.output/"summary.json").write_text(json.dumps(dict(scripted_baseline=True,
                learned_policy=False, independent_acceptance=False,cases=rows),indent=2))
            print(json.dumps(record),flush=True)
    lines=["# 显式脚本示范：R2物理可行性对照","",
        "仅作物理对照和训练标签来源，不是学习策略成绩。相同24维动作、限速、QP、关节余量、负载和执行器力矩上限；手指名义领先量0.15rad。", "",
        "| 条件 | 完整执行 | 任务达标 | 最差运动RMSE | 最差保持峰值 |",
        "| --- | --- | --- | --- | --- |"]
    for name,_,_ in plans:
        group=[r for r in rows if r['condition']==name]
        lines.append(f"| {name} | {sum(r['complete'] for r in group)}/{len(group)} | {sum(r['passed'] for r in group)}/{len(group)} | {max(r['rmse_deg'] for r in group if r['case']!='hold'):.4f}° | {max(r['peak_error_deg'] for r in group if r['case']=='hold'):.4f}° |")
    (args.output/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=Path("configs/r22_demonstration_bootstrap_v3.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--resume",action="store_true")
    run(parser.parse_args())
