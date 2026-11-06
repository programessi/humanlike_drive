"""Log physical acceptance metrics, explicitly separate from RL rewards."""
import argparse
import json
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


def read(path):
    try: return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError): return {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--sentinel',type=Path)
    parser.add_argument('--logdir',type=Path,required=True)
    args=parser.parse_args()
    writer=SummaryWriter(str(args.logdir));seen={}
    writer.add_text('method','Physical closed-loop acceptance of supervised structured actor. These curves are not PPO reward or training loss.',0)
    while True:
        ledger=read(args.ledger)
        dirs=([args.sentinel] if args.sentinel else [])+[Path(s['output']) for s in ledger.get('stages',[])]
        for directory in dirs:
            result=read(directory/'summary.json')
            cases=result.get('cases',[])
            count=len(cases)
            if not count or seen.get(str(directory))==count:continue
            seen[str(directory)]=count;tag=directory.name
            writer.add_scalar(tag+'/pass_fraction',sum(c['passed'] for c in cases)/count,count)
            writer.add_scalar(tag+'/hard_failures',sum(c.get('hard_failure',False) for c in cases),count)
            writer.add_scalar(tag+'/min_arm_margin_deg',min(c['min_arm_margin_deg'] for c in cases),count)
            errors=[c.get('motion_rmse_deg',c.get('rmse_deg')) for c in cases if c['complete']]
            stops=[p for c in cases for p in c.get('stop_peaks_deg',[c.get('tail_peak_deg')]) if p is not None]
            if errors:writer.add_scalar(tag+'/worst_complete_motion_rmse_deg',max(errors),count)
            if stops:writer.add_scalar(tag+'/worst_stop_physics_peak_deg',max(stops),count)
            writer.flush()
        if ledger.get('status','').startswith(('complete_','rejected_','sentinel_rejected')):break
        time.sleep(20)
    writer.add_text('final_status',ledger.get('status',''),len(seen));writer.close()


if __name__=='__main__':main()
