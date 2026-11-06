"""Report every demonstration/DAgger round, including regressions."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
config=json.loads((args.run/'config.json').read_text())
index=json.loads((args.run/'development_index.json').read_text())
rows=[]
for entry in index:
    summary=json.loads((args.run/'development'/f"round_{entry['round']:02d}"/'summary.json').read_text())
    cases=summary['cases']
    rows.append(dict(**entry,sha256=hashlib.sha256((args.run/f"round_{entry['round']:02d}.zip").read_bytes()).hexdigest(),
        hold_peak_deg=max(r['peak_error_deg'] for r in cases if r['case']=='hold'),
        max_motion_rmse_deg=max(r['rmse_deg'] for r in cases if r['case']!='hold'),
        max_tail_peak_deg=max((r['tail_peak_deg'] for r in cases if r['tail_peak_deg'] is not None),default=None),
        min_arm_margin_deg=min(r['min_arm_margin_deg'] for r in cases),
        collision_peak_n=max(r['collision_peak_n'] for r in cases),
        free_force_peak_n=max((r['free_hand_peak_force_n'] or 0 for r in cases)),
        qp_failures=sum(r['qp_failures'] for r in cases),
        failures=[r for r in cases if not r['passed']],
        baseline_improvements=summary['baseline_improvements']))
(args.output/'summary.json').write_text(json.dumps(dict(run=str(args.run),config=config,rounds=rows),indent=2)+'\n')
lines=['# 示范辅助统一网络：逐轮实测','',
    '训练使用显式脚本标签和BC/DAgger；评价仅执行网络全部24维输出。特权教师，尚未训练历史学生。','',
    '| 轮次 | 累计交互样本 | 采集时专家概率 | 完整 | 达标 | 最差运动RMSE | 最差保持峰值 |',
    '| --- | --- | --- | --- | --- | --- | --- |']
for row in rows:
    lines.append(f"| {row['round']} | {row['samples']} | {row['expert_probability']} | {row['complete']}/{row['total']} | {row['passes']}/{row['total']} | {row['max_motion_rmse_deg']:.6f}° | {row['hold_peak_deg']:.6f}° |")
lines+=['','模型哈希、各轮失败、基线改善率、关节余量和QP计数见summary.json。best按训练时冻结规则保存，不能用final代替。开发集通过不代表独立验收或整个R2通过。']
(args.output/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'output':str(args.output),'rounds':len(rows)}))
