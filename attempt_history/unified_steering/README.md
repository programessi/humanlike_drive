# 统一双臂双手方向盘跟踪

从已建立的抓握开始，研究一个策略在有限工作空间与轮轴负载下，完成带时间约束的轮角跟踪、停止保持、自主接管和双向连续操作。上游提供参考轨迹及有限预览；本项目不处理道路视觉、驾驶规划和初始抓取。

**当前正式策略：v121结构化24维腕手统一策略＋IK/QP，示范辅助学习。新版R0–R2已按90°双手及180°提前接管范围验收通过：同一权重开发/独立/回归合计624/624，安全硬失败0。权重与加载说明见[正式模型](runs/driving_r2_frozen_v121/README.md)，[视频](results/DRIVING_R0_R2_ACCEPTANCE_V121/videos/index.html)及[验收报告](results/DRIVING_R0_R2_ACCEPTANCE_V121/README.md)已保留。360°/540°、原固定上方手风格及学生部署未验证；唯一总计划见[RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)。**

当前正式模型读取152维actor观测及上游序列范围2维，输出24维腕手动作。旧PPO及关节控制代码保留作对照。历史检查点、训练数据和中间轨迹已按用户要求清理；最终指标、视频和必要测试夹具保留，见[产物索引](results/README.md)。

## 组织

采用 `src` 包布局，将运行配置、实验方案、审阅结果与原始输出分开。以下列出已经实现的关节路线模块；新腕手教师迁移继续归入本项目，具体边界以最新计划为准。

```text
unified_steering/
  src/unified_steering/
    contract.py          版本化输入输出定义、关节顺序
    audit.py             场景快照、结构检查、保持物理诊断
    wheel_load.py        固定机械零位的回正负载
    support.py           单手支持、释放、有限幅反馈脚本对照
    clearance.py         现有碰撞几何距离检查
    support_report.py    图表归档、有效初态筛选
    initialization.py    带校验和的快照加载、离线双手初态准备
    simulation.py        复用审计过的物理、接触与安全检查
    execution.py         26维有限速/领先量执行器
    reference.py         有限速加速度的累计轮角时间参考
    observations.py      152/172维输入及固定归一化
    rewards.py           分项奖励与可配置权重
    env.py               Gymnasium reset/step
    evaluation.py        独立评价指标
    validate_env.py      保持回归、批量运行与吞吐报告
    models.py            双分支GRU actor、独立critic
    distributions.py     tanh Gaussian采样与稳定概率修正
    workers.py           并行物理、最终观测与reset分离
    rollout.py           时序数据、GRU历史与GAE
    ppo.py               时序minibatch、截断目标与梯度更新
    checkpoint.py        模型/优化器/配置/RNG保存恢复
    train.py             训练与日志入口
    evaluate.py          固定开发测试、策略与固定目标对照
    telemetry.py         可选500 Hz物理字段
    tracing.py           逐步动作/接触/关节诊断与NPZ
    snapshots.py         仅诊断使用的精确状态分支
    monitoring.py        周期评价、编号checkpoint及保持保护
    diagnose.py          冻结复评、配对噪声和方向响应入口
    disturbance.py       声明式有界轮轴脉冲，默认关闭
    hold_baseline.py     3/10秒保持、脉冲及配对探索评价
  configs/
    s00.yaml             可运行的诊断配置
    s00_support.yaml     单手/释放/停放随机化诊断配置
    stage_a.yaml         候选网络与PPO参数
    m1.yaml              可运行的最小训练环境配置
    m2_smoke.yaml        可运行的recurrent PPO工程试训配置
    m30_diagnosis.yaml   冻结策略的10种子诊断协议
    m30_monitor_smoke.yaml 1024步评价设施集成测试
    m31_hold_baseline.yaml 4.1冻结保持基线及预声明门槛
  scripts/run.sh         本项目独立运行入口
  tests/                 回正负载物理性质与无接触回正检查
  assets/                场景快照、来源与外部模型依赖说明
  docs/                  架构约定、实验次序和通过门槛
  results/               可审阅的实验结论与下一步决策
  outputs/               原始 CSV、完整配置、summary.json（不入 Git）
  runs/                  训练checkpoint与原始日志（不入Git）
  pyproject.toml
```

- [最新研究与实施计划](docs/RESEARCH_PLAN.md)：新版R0驾驶可行域审计→R1双手90°往返→R2单手主导180°往返→R3接管重抓360°→R4扩展540°→R5历史学生。
- [已实现的关节架构与接口（冻结对照）](docs/ARCHITECTURE.md)
- [M1训练环境调用、量纲与终止约定](docs/TRAINING_ENVIRONMENT.md)
- [M2网络、训练闭环与复现命令](docs/TRAINING_LOOP.md)
- [M3-0诊断日志与周期评价调用](docs/DIAGNOSTICS.md)
- [历次方案全面对照](docs/SCHEME_COMPARISON.md)
- [实验结果入口](results/README.md)
- [场景来源与限制](assets/README.md)
- [回正负载模型与配置](docs/WHEEL_LOAD.md)
- [单手支持诊断协议](docs/SINGLE_HAND_PROTOCOL.md)与[已完成的S00-B结果](results/S00_SUPPORT.md)。

## 已有物理与工程工具的复现

以下命令用于已有场景、关节环境和工程设施的复现，不是新腕手主线的训练入口。当前驾驶主线按最新计划推进；`configs/r1_r2_v1.json`属于旧小角度协议。

从父目录运行：

```bash
# 仅首次：冻结已有场景。只读取 XML/外部网格，不调用旧控制器或策略。
bash unified_steering/scripts/run.sh -m unified_steering.audit prepare \
  --source ../manipulation/assets/regrasp_stage1/wheel.xml

# 每次使用一个新输出目录，防止覆盖结果。
bash unified_steering/scripts/run.sh -m unified_steering.audit run \
  --output outputs/s00_fixture_v1
```

场景快照存在时，`prepare` 会拒绝覆盖；重复诊断只运行 `run`，并使用新的输出目录。

本机默认 Python：`/home/nubot/anaconda3/envs/mujoco/bin/python`；可通过 `STEERING_PYTHON` 指定其他环境，或 `pip install -e ./unified_steering` 后运行模块。迁移网格时，在首次 `prepare` 前设置 `H1_MESH_DIR`。

当前环境验收（在本目录运行）：

```bash
bash scripts/run.sh -m unittest discover -s tests -v
bash scripts/run.sh -m unified_steering.validate_env --output outputs/m1_environment_new_run
```

历史 M2 工程试训入口（仅供复现，不作为当前主线启动命令；输出目录使用新名字）：

```bash
bash scripts/run.sh -m unified_steering.train --output runs/m2_smoke_new_run
```

诊断只在reset时设置初始状态，重力/偏置补偿包含在有限执行器输出内；轮轴按实际累计角度施加回正弹簧力矩，黏性阻尼与干摩擦由关节物理计算。原始保持诊断无轮角反馈；新的单手协议另提供显式`--feedback`脚本对照，并记录reset辅助夹具撤除。该反馈不进入未来策略执行层。负载为仿真起始值，不是硬件标定结果。通过诊断不代表已经学会跟踪，也不构成完整碰撞安全证明。

## 已实现的小角度腕手训练入口

在本目录运行；各训练必须使用新输出目录：

```bash
bash scripts/run.sh -m unified_steering.structured.train --stage R1 \
  --resume ../manipulation/runs/regrasp_stage1/control_v3/block_04_A10/teacher.zip \
  --output runs/r1_new_run --steps 131072
```

上述命令中的`--stage R1/R2`仍对应旧小角度实现，不代表新版90°/180°课程已实现。接口见[STRUCTURED_INTERFACE.md](docs/STRUCTURED_INTERFACE.md)。旧版R0、R1、R2已通过：同一冻结示范辅助特权教师独立名义72/72、筛选初态168/168，共240/240。见[R2验收报告](results/R2_ACCEPTANCE_V4/README.md)和[总计划](docs/RESEARCH_PLAN.md)。模型位于`runs/r2_frozen_v4/teacher.zip`。训练已结束，TensorBoard http://localhost:6006 可查看曲线。新版R0已分段验证90°名义往返的物理入口，见[审计报告](results/DRIVING_R0_AUDIT_V1/README.md)；本轮未更新网络或场景，新版R1网络90°及R2动态180°仍未通过，R3换手、R4单侧540°、R5历史学生尚未开始。
