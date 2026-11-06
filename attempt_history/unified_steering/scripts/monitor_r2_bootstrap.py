"""Publish measured BC/DAgger rounds to the existing TensorBoard server."""
import argparse
import json
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
args=parser.parse_args()
seen=set()
with SummaryWriter(str(args.run/'tensorboard'/'BC_DAgger')) as writer:
    for _ in range(720):
        config=json.loads((args.run/'config.json').read_text())
        path=args.run/'development_index.json'
        if path.exists():
            rows=json.loads(path.read_text())
            for row in rows:
                if row['round'] in seen:continue
                step=row['samples']
                for key in ('complete','passes','total','score','passed'):
                    writer.add_scalar('development/'+key,float(row[key]),step)
                for key in ('bc_initial_loss','bc_final_loss','expert_probability'):
                    writer.add_scalar('training/'+key,row[key],step)
                seen.add(row['round'])
                writer.flush()
        if config['status']!='training':break
        time.sleep(10)
