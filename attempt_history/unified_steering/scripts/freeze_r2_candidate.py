"""Freeze only after two complete screened development passes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
config=json.loads((args.run/'config.json').read_text())
protocol=json.loads((args.run/'protocol.json').read_text())
index=json.loads((args.run/'development_index.json').read_text())
if protocol['parking']!='screened' or not config.get('qualified') or len(index)<2 or not all(r['passed'] and r['total']==168 for r in index[-2:]):
    raise ValueError('Require two consecutive full168-case screened development passes')
source=args.run/'qualified_development.zip'
args.output.mkdir(parents=True,exist_ok=False)
shutil.copy2(source,args.output/'teacher.zip')
for name in ('config.json','protocol.json','development_index.json'):
    shutil.copy2(args.run/name,args.output/('source_'+name))
digest=hashlib.sha256(source.read_bytes()).hexdigest()
config.update(parent_training_run=str(args.run.resolve()),status='frozen_requires_independent_acceptance',r2_passed=False,
    frozen_model_sha256=digest,interface='structured-wrist-teacher-v4-separate-hand-lead')
(args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
acceptance=dict(model=str((args.output/'teacher.zip').resolve()),model_sha256=digest,
    nominal_seeds=[1207,1217,1227],screened_seeds=[1307,1317,1327],
    nominal_cases=72,screened_cases=168,command_protocol='both-sequence-signs-v2',
    criteria='Each case complete, motionRMSE<=1deg and last3s peak<=1deg; hold peak<=1deg; free hand force<=0.2N; standard progress and baseline30% improvement gates.',
    train_after_test=False,script_used_at_evaluation=False)
(args.output/'acceptance_protocol.json').write_text(json.dumps(acceptance,indent=2)+'\n')
print(json.dumps(acceptance))
