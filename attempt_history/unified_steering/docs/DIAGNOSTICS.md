# M3-0：只读诊断与周期开发评价

本文记录现有关节策略的诊断工具。当前研究阶段与迁移安排见[最新研究计划](RESEARCH_PLAN.md)，旧 M3 编号仅用于定位历史实验。

历史结论见[清理后的摘要](../results/README.md)。旧 M2/M3 检查点、诊断轨迹和归档结果已清理；本页只描述源码接口，依赖旧权重或旧归档的命令不能直接执行，也不代表后续课程或阶段验收已实现。

## 冻结对照

```bash
bash scripts/run.sh -m unified_steering.diagnose \
  --config configs/m30_diagnosis.yaml --output outputs/m30_diagnosis_new
```

配置固定checkpoint、10个配对种子、3秒保持、探索尺度和方向锚点。依次执行旧15例复评、40次随机保持及条件触发的组合组、两actor×三个锚点×正负方向的短回放。若旧RMSE差超过10⁻⁶度、源checkpoint改变或冻结参数发生变化，运行报错。输出目录拒绝覆盖。

方向锚点保存MuJoCo积分状态、warmstart、GRU隐藏状态、部署输入及观测历史。物理分支从同一状态恢复，以平滑参考回放；瞬时±5°误差仅检查输入敏感性，不直接送给物理执行。快照工具仅用于因果诊断，不用于训练期自动修复。

## 日志

环境配置`diagnostics: true`启用`info['diagnostics']`；默认关闭，不改变观测和奖励。每个控制调用有逐关节动作处理计数、接触力/力矩和计数、主动/被动关节限位详情，以及每个2 ms物理子步的数组。4.0归档为21字段；4.1的`steering-trace-v2`追加扰动与总轮轴力矩，现为23字段，原回正字段单独表示回正。按`physics_fields`读取列，不按硬编码列数读取。字段见`telemetry.PHYSICS_FIELDS`。

`TraceRecorder`另存每个50 ms控制调用的152维输入、26维潜变量均值/标准差/样本、原始和执行动作、目标/误差/驱动力、39维qpos/qvel及500 Hz物理数据。NPZ无需pickle加载；元数据为JSON字符串。摘要与NPZ同名、后缀`.summary.json`。

逐轴处理类型：`arm_lead`目标领先量投影、`arm_bound`关节目标边界投影、`arm_slew`臂目标速度投影、`hand_slew`手目标限速。计数按物理tick累计，同tick多原因可以重叠。正常有效tanh动作的臂速度本来已受限，`arm_slew`通常为零。最长连续时间为该回合范围，包括方向分支之前的预热；局部频率仅计当前记录片段。

旧`execution_interventions`保留原语义：每个发生任一臂目标投影的物理tick计一次，评价除以总物理tick；新逐轴/分原因频率不能直接相加，因为可能重叠，也不能把手目标限速计入旧指标。手轮接触采用各接触力范数求和，避免净力抵消；双手都≤0.2 N持续100 ms只是低接触力代理，不证明几何分离或抓握承载能力。

独立确定性评价也可导出日志：

```bash
bash scripts/run.sh -m unified_steering.evaluate \
  --checkpoint runs/m2_smoke_seed7_v1/final.pt \
  --output outputs/m30_final_new --diagnostics
```

提前失败会令未完成的末段保持指标为`null`、成功为`false`；已观察前缀RMSE保留，但不能按完整成绩排序。完整执行旧案例的指标未改变。

## 周期评价与保存

训练配置增加：

```yaml
monitoring:
  evaluation_every_steps: 10240
  retain_step_checkpoints: true
  diagnostics: false
```

间隔必须为`num_envs × rollout_steps`的整数倍。启用后在初始及对应更新之后使用独立环境执行旧5例固定开发评价，保留`step_000000000.pt`等编号文件及评价结果。`retain_step_checkpoints`还可在原有checkpoint更新间隔保留编号文件。周期评价保存/恢复全局Python、NumPy、Torch RNG和模型模式，不消耗训练环境状态。

`development.jsonl`与`development_index.json`记录候选资格、评分、最佳步数。资格要求旧5例全完成、保持RMSE≤初始锚点+0.5°且漂移≤2°；再按跟踪成功率、最差及平均运动RMSE选`best.pt`。`latest.pt`继续保存最新训练状态。初始锚点仅为本运行基线，后续M3合格保持集/独立验收尚需实施，当前没有完整阶段合格checkpoint。

`--skip-evaluation`仅跳过训练末尾三套对照，不关闭配置启用的周期评价。训练墙钟时间包含同步开发评价；比较吞吐时需保持评价设置一致。旧M2配置未启用该设施，仍保持旧行为。

设施短测命令：

```bash
bash scripts/run.sh -m unified_steering.train \
  --config configs/m30_monitor_smoke.yaml \
  --output runs/m30_monitor_new --skip-evaluation
```

该配置仅1024步、评价间隔512，不作为M3技能结果。默认日志开关关闭；不建议在常规训练每个worker保存全部高频日志，先在固定评价上按需启用。

## 4.1冻结保持与脉冲入口

```bash
bash scripts/run.sh -m unified_steering.hold_baseline \
  --config configs/m31_hold_baseline.yaml --output outputs/m31_hold_new
```

环境新增可选`wheel_disturbance`，默认无脉冲。当前评估入口只允许不超过0.05 N·m、时刻对齐物理tick且互不重叠的脉冲。它通过原有轮轴外力矩入口叠加，不改名义弹簧/阻尼/摩擦和执行器；actor不接收脉冲真值。历史配置模板见[源码配置](../configs/m31_hold_baseline.yaml)，其中旧权重和归档路径已失效。没有有辨识力的扰动时不会宣称抗扰恢复通过；额外外扰不作为当前主线的前置门槛。
