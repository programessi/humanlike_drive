"""Freeze and evaluate a learned R2 candidate; no training or script execution."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import torch

from unified_steering.structured.env import TaskConfig
from unified_steering.structured.evaluate import load_teacher,suite
from unified_steering.structured.parking import SCREENED_RESETS,SupportTeacherEnv

parser=argparse.ArgumentParser()
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--seeds',type=int,nargs='+',required=True)
parser.add_argument('--parking',choices=('nominal','screened'),default='nominal')
parser.add_argument('--independent',action='store_true')
parser.add_argument('--workers',type=int,default=2)
parser.add_argument('--baseline-root',type=Path)
args=parser.parse_args()
torch.set_num_threads(1)
args.output.mkdir(parents=True,exist_ok=False)
source=json.loads((args.model.parent/'config.json').read_text())
task=source['task']
config=TaskConfig(hand_action_mode=task['hand_action_mode'],hand_leak_alpha=task['hand_leak_alpha'],
    hand_joint_lead_rad=task.get('hand_joint_lead_rad'),angle_reward_sigma_deg=task['angle_reward_sigma_deg'])
metadata=dict(model=str(args.model.resolve()),sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
    seeds=args.seeds,parking=args.parking,independent_acceptance=args.independent,
    task=asdict(config),scripted_execution=False,trained_here=False,status='running')
(args.output/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
model=load_teacher(args.model)
plans=[('nominal',{},('both','left','right'))]
if args.parking=='screened':plans += [(name,options,('left','right')) for name,options in SCREENED_RESETS.items()]
result=dict(cases=[],conditions={},baseline_improvements={},passed=True,independent_acceptance=args.independent)
for name,options,supports in plans:
    existing=args.baseline_root/name/'summary.json' if args.baseline_root else None
    if existing and existing.exists():
        baseline=json.loads(existing.read_text())
        if baseline['task']!=asdict(config) or sorted({r['seed'] for r in baseline['cases']})!=sorted(args.seeds):
            raise ValueError('Reused baseline configuration/seeds differ')
        (args.output/'baseline').mkdir(exist_ok=True)
        shutil.copy2(existing,args.output/'baseline'/f'{name}.json')
    else:
        baseline=suite(None,config,seeds=args.seeds,supports=supports,output=args.output/'baseline'/name,
            env_class=SupportTeacherEnv,reset_options=options,workers=args.workers,
            include_reverse_sequence=True,independent=args.independent)
    part=suite(model,config,seeds=args.seeds,supports=supports,output=args.output/name,
        baselines=baseline,env_class=SupportTeacherEnv,reset_options=options,workers=args.workers,
        include_reverse_sequence=True,independent=args.independent)
    result['conditions'][name]=part;result['cases']+=part['cases']
    result['baseline_improvements'].update({name+'/'+key:value for key,value in part['baseline_improvements'].items()})
    result['passed']=result['passed'] and part['passed']
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
metadata['status']='passed' if result['passed'] else 'failed'
(args.output/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
lines=['# R2冻结网络评价','',f"独立验收：{args.independent}；模型SHA256：`{metadata['sha256']}`。",'',
    '统一网络输出全部24维动作；无脚本、动作覆盖或训练更新。','',
    '| 条件 | 完整执行 | 达标 | 最差运动RMSE | 最差保持峰值 |',
    '| --- | --- | --- | --- | --- |']
for name,part in result['conditions'].items():
    rows=part['cases']
    lines.append(f"| {name} | {sum(r['complete'] for r in rows)}/{len(rows)} | {sum(r['passed'] for r in rows)}/{len(rows)} | {max(r['rmse_deg'] for r in rows if r['case']!='hold'):.6f}° | {max(r['peak_error_deg'] for r in rows if r['case']=='hold'):.6f}° |")
lines+=['',f"总结果：{result['passed']}。逐例、基线改善率、初态与执行裁剪见summary.json；本结果属于特权教师，不表示换手或学生部署通过。"]
(args.output/'README.md').write_text('\n'.join(lines)+'\n')
print('VALIDATION '+json.dumps(dict(passed=result['passed'],cases=len(result['cases']),output=str(args.output))),flush=True)
