"""Paired evaluations with independent randomization seeds."""
import argparse
import contextlib
import json
from pathlib import Path
import numpy as np
from .evaluate import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("outputs/benchmark"))
    p.add_argument("--seeds",type=int,nargs="+",default=[101,102,103,104,105])
    p.add_argument("--amplitude",type=float,default=45)
    p.add_argument("--seconds",type=float,default=30)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results={}
    for label,model in (("baseline",None),("ppo",args.model)):
        summaries=[]
        for seed in args.seeds:
            path=args.output/f"{label}_{seed}"
            with (args.output/f"{label}_{seed}.log").open("w") as log,contextlib.redirect_stdout(log):
                summary=run(path,model,mode="sine",amplitude=args.amplitude,seconds=args.seconds,seed=seed,randomize=True)
            summaries.append(summary)
            print(label,seed,"RMSE",round(summary["rmse_deg"],3),"failed",summary["early_failure"],flush=True)
        results[label]={"episodes":len(summaries),"failures":sum(s["early_failure"] for s in summaries),
                        "mean_rmse_deg":float(np.mean([s["rmse_deg"] for s in summaries])),
                        "std_rmse_deg":float(np.std([s["rmse_deg"] for s in summaries])),
                        "mean_contact_fraction":float(np.mean([s["any_hand_contact_fraction"] for s in summaries])),
                        "max_force_N":float(max(s["peak_contact_force_N"] for s in summaries))}
    (args.output/"paired_summary.json").write_text(json.dumps(dict(seeds=args.seeds,amplitude_deg=args.amplitude,seconds=args.seconds,results=results),indent=2))
    print(json.dumps(results,indent=2))


if __name__=="__main__":main()
