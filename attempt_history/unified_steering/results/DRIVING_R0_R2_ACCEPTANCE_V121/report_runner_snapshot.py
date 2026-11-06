"""Build an acceptance report only after every predeclared gate passes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def metrics(result):
    cases=result['cases']
    errors=[c.get('motion_rmse_deg',c.get('rmse_deg')) for c in cases if c['complete']]
    stops=[p for c in cases for p in c.get('stop_peaks_deg',[c.get('tail_peak_deg')]) if p is not None]
    return dict(count=len(cases),passes=sum(c['passed'] for c in cases),hard=sum(c.get('hard_failure',False) for c in cases),
        worst_rmse_deg=max(errors),worst_stop_deg=max(stops),full_physics_peak_deg=max(c['peak_error_deg'] for c in cases),min_margin_deg=min(c['min_arm_margin_deg'] for c in cases),
        collision_peak_n=max(c['collision_peak_n'] for c in cases),qp_failures=sum(c['qp_failures'] for c in cases))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ledger=json.loads(args.ledger.read_text());version=args.ledger.stem.rsplit('_',1)[-1]
    if ledger['status']!='complete_all_gates_passed':raise RuntimeError('All physical gates must complete before acceptance')
    model=Path(ledger['model']);digest=hashlib.sha256(model.read_bytes()).hexdigest()
    assert digest==ledger['sha256']
    records={}
    for stage in ledger['stages']:
        result=json.loads((Path(stage['output'])/'summary.json').read_text())
        assert stage['passed'] and result['passed']
        expected=240 if stage['name']=='old240_retention' else len(json.loads(Path(stage['protocol']).read_text())['cases'])
        assert result['count']==expected
        meta_path=Path(stage['output'])/('protocol.json' if stage['name']=='old240_retention' else 'metadata.json')
        meta=json.loads(meta_path.read_text());assert meta.get('sha256',meta.get('model_sha256'))==digest
        if stage['name']=='independent':assert result['independent'] and meta['independent']
        records[stage['name']]=dict(output=stage['output'],**metrics(result))
        assert records[stage['name']]['hard']==0
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'evidence.json').write_text(json.dumps(dict(model=str(model),sha256=digest,stages=records,scope='finite nominal initial-grasp conditions; <=90deg and nominal180deg early-regrasp variant, 5deg/s'),indent=2)+'\n')
    rows=[]
    names=dict(development_a='R2开发A',development_b='R2开发B',consumed550_stress='已消耗550xx压力测试',r1_retention='同权重R1回归',old240_retention='同权重旧240回归',independent='R2独立验收')
    for name,m in records.items():
        rows.append(f"| {names[name]} | {m['passes']}/{m['count']} | {m['worst_rmse_deg']:.6f}° | {m['worst_stop_deg']:.6f}° | {m['min_margin_deg']:.6f}° | {m['hard']} |")
    links='\n'.join(f"- [{names[name]}逐例报告](../../{m['output']}/summary.json)" for name,m in records.items())
    report=f'''# 新版R0–R2：当前有限范围验收通过

R0已分段审计90°及180°提前接管变体的物理入口；R1原冻结策略通过90°验收；本次同一结构化统一actor通过R2及基础能力保留测试。当前180°方案会提前释放、接管，不等同于原“上方手固定握至180°”风格。360°/540°、任意在线改令和部署学生尚未验证。

冻结模型：[{model.name}](../../{model})  
SHA256：`{digest}`。

| 测试 | 通过 | 最差运动RMSE | 最差停止末段峰值 | 最小实测臂余量 | 安全硬失败 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

独立套件最大全程瞬态物理步误差为{records['independent']['full_physics_peak_deg']:.6f}°，已单独披露；它与停止稳定精度是不同指标。

R2覆盖0→±180°→0和双方向连续完整往返，短保持3秒、长保持10秒，名义及reset预转2°两个有限初态条件。参考峰值速度5°/s、加速度上限40°/s²，外部参考时钟不因换手暂停；累计轮角不取模。运动RMSE≤1°、每次停止末段3秒的物理步峰值≤1°；瞬态全程峰值另报，不表示所有时刻误差都≤1°。

回正刚度0.3Nm/rad、手执行器0.35Nm上限、真实2°臂余量、0.02rad关节越界容差、1N非法碰撞与QP失败拒绝规则保留。执行为35mm命令避碰间隔/100mm提前激活和4°臂命令预留；当前轮轴仍±450°止挡。没有执行中绕过24维动作改写物理状态、执行器命令或外部时钟。

## 实际模型与数据边界

这是一个checkpoint内的监督结构化混合actor：合格R1位姿网络、大角度归一化径向基函数、神经内部腕部锚点门和上游指令范围混合门，统一输出24维双腕/双手动作，经有限执行器及IK/QP实现。不是原168→24 PPO模型或单一共享MLP联合训练成功，也不是PPO自主发现换手。

R2基函数11,297个中心来自本项目48361训练种子的双向成功仿真序列；锚点神经门7→32→32→2使用v77/v89成功训练示范，训练抽样12,000条；指令范围门2→16→1由6,000条合法上游序列范围监督拟合。R1位姿层和手控制沿用合格教师，完整来源见冻结config及历代报告。推理不读取示范文件、专家动作、手部阶段或模板ID，仿真接触真值和旧privileged字段不被此模型读取。

输入为actor152反馈/参考及上游已提供的角度序列min/max两维；记忆由策略自身更新。内部腕部锚点更新不重置机器人、执行器、回合或参考时间；手部抓握基准在整条序列中保留，避免重复叠加预紧。R2内部相位补偿上限40°，小角度分支仍30°。当前只验证≤90°和名义180°任务，二者之间的范围、跨包络指令变化、连续初态及实物鲁棒性不能由本报告推出。

## 验证与复现

先开发A36、开发B36、已消耗550xx整组120例压力测试，再做当前执行器下的R1精确72例和旧240例回归，全部合格后冻结同一权重和代码，最后执行此前未用于调参的560xx独立120例。独立评价期间未更新权重。两个开发集合、旧R1和旧240都是已用工程案例；120例含配对种子，不作为120个独立伯努利样本或真实总体成功率的置信保证。

126项工程检查通过：[日志](../../outputs/driving_full_tests_{version}.log)。v119虽通过两轮开发和保留回归，仍因独立验收硬失败被拒绝；当前候选修复了跨轮抓握预紧累积，并将R2内部运动相位补偿上限由30°改为40°，没有放松真实物理/执行器/IK边界。两条原失败种子与双方向连续诊断4/4通过。包括v119在内的失败探索保留在[唯一计划](../../docs/RESEARCH_PLAN.md)，没有从研究记录中删除。

{links}
- [机器可读汇总](evidence.json)
- [冻结配置与源代码清单](../../{model.parent}/manifest.json)
- [独立序列物理曲线](figures/physical_sequences.png)
- [完整执行顺序](../../{args.ledger})

重跑单个大角度套件：`PYTHONPATH=src python -m unified_steering.structured.driving_learning evaluate --model {model} --protocol configs/driving_r2_independent_{version}.json --output results/NEW_UNIQUE_DIRECTORY --workers 4`。560xx在本次验收后已使用，未来调参后重跑只能称回归，不能重新称新留出。
'''
    (args.output/'README.md').write_text(report)
    shutil.copy2(__file__,args.output/'report_runner_snapshot.py')
    print(args.output/'README.md')


if __name__=='__main__':main()
