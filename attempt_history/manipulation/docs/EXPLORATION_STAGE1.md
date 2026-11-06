# 集中式腕手 GRU：第一阶段探索

本阶段从已知初始抓握开始，探索无上游、无视觉、无换手的小角度保持和双向跟踪。它是独立的新策略接口，不覆盖已有的 18 维接触残差模型，也不覆盖 baseline 项目。

## 控制与网络

内部目标序列 → 确定性平滑参考及 0.5 s 预览 → MLP(256,128) → GRUCell(128) → 24 维动作 → 联合 14 变量 IK/QP 和手部目标 → 限幅隐式 PD → 物理接触 → 实测轮角。

部署 actor 不读取仿真接触、力、摩擦或从动指节角。训练使用独立的 `[512,256,128]` critic，输入 actor 观测及 54 维仿真特权量。策略 20 Hz、臂执行 50 Hz、物理 500 Hz。CPU 单线程、单环境，训练使用与旧 manipulation 共享的进程锁。

执行层没有轮角误差反馈运动控制器，没有随目标角移动的圆弧腕部参考。初始抓握仍由原有静态 IK 和物理闭合准备建立。

## 单步 actor 观测：140 维

| 索引（含首、不含尾） | 内容 | 数值处理 |
| --- | --- | --- |
| 0:26 | 双臂 14 + 双手 12 个主动驱动位置 | 弧度；实物需标定驱动读数映射 |
| 26:52 | 位置差分滤波得到的主动驱动速度 | 除以 5 rad/s |
| 52:78 | 当前关节目标减实测主动驱动位置 | 除以 0.15 rad |
| 78:102 | 上一步策略动作 | [-1,1] |
| 102:120 | 双腕相对轮轴的位置与旋转前两列 | 位置除以 0.2 m；每腕 9 维 |
| 120:122 | 实测轮角、轮角差分滤波速度 | rad、rad/s |
| 122:128 | 当前及未来 0.1/0.2/0.3/0.4/0.5 s 参考角减当前测得轮角 | 除以 10° |
| 128:129 | 当前参考角速度 | 除以 20°/s |
| 129:140 | 已知安装中心、安装旋转前两列、轮辋半径、截面半径 | 中心/0.8 m，半径/0.14 m，截面半径/0.014 m |

未来量来自已知的指令序列，不是未来真实状态。FK 只使用主动驱动读数和固定机构几何，不能读取真实不可得的模拟从动关节。当前没有把前 8 帧重复输入：单步观测交给 GRU，运行端负责保留并在 episode 开始清空记忆。

## 动作：24 维

- 0:6：左腕三维平移及三维旋转命令。
- 6:12：右腕三维平移及三维旋转命令。
- 12:18、18:24：左、右手各六个驱动的绝对目标。

Actor 输出高斯 latent，经 tanh 得到 [-1,1] 动作。腕部命令按实际 robot_dt 积分到独立位姿目标，最大线速度 0.04 m/s、最大角速度 25°/s，使用 0.1 s 一阶滤波。目标限制在初始位置各轴 ±0.06 m、初始姿态 40° 内，并限制对实测腕部的超前为 12 mm/10°。这些是本阶段探索范围，不是 ±90°重抓的最终工作空间。

手部动作采用分段线性绝对映射：-1 为有效下限，0 为名义闭合，+1 为有效上限，经过手部速度/加速度和关节超前限制。它不再是仅能微调闭合的 ±0.15 rad 残差。

联合 IK 是对两腕共同求解的 14 变量有界最小二乘问题；接近预设肘部/掌部/躯干几何对时增加线性距离约束。只有少量保守几何对，不能作为完整手指/驾驶室避碰保证。物理接触、异常载荷及 IK 失败另行记录和终止。

## 训练边界与奖励

阶段为 `hold` → `reach3` → 后续 `reach10`。先进行小规模 seed 7 试训，不能把不同评估初态种子称为多个训练种子。

保持与小角度跟踪阶段允许利用已知轮轴几何做**奖励塑形**：把当前目标轮角对应的原抓握区腕位姿与实际位姿比较，提供稠密几何奖励。这些位姿不送入执行器、不替策略转动方向盘，也没有人类运动轨迹。未来重抓阶段必须调整这项固定抓握奖励，否则会阻碍释放和重定位。

其他奖励包含真实轮角跟踪、包络/接触、动作平滑以及异常载荷惩罚。训练优化使用原始奖励的 0.1 倍，CSV 和评价记录原始奖励。第一阶段允许奖励双手保持；这不是未来所有阶段始终要求两手接触的设计。

物理准备时随机改变闭合目标和稳定时间，之后由动力学生成接触。动力学、摩擦、执行响应、观测和延迟沿用已有随机化。没有直接赋值接触力，也没有在策略执行中写入轮角。

PPO 使用 256 步 rollout、64 步序列、每批 4 个序列、4 个 epoch。每段携带 rollout 时的 GRU 初始状态，episode 内保留记忆、跨 episode 重置。时间截断使用 terminal observation 的价值 bootstrap，但不把下一回合的 GAE 传播回上一回合。熵项是 latent 高斯熵，动作概率含 tanh 变换的 Jacobian。

## 运行与产物

在 manipulation 项目目录执行，训练和评价顺序运行。每次提供新的输出目录。

```bash
bash scripts/run.sh -m wheel_manipulation.exploration.train \
  --stage hold --steps 8192 --seed 7 \
  --output runs/exploration_stage1/hold_s7_pilot

bash scripts/run.sh -m wheel_manipulation.exploration.train \
  --stage reach3 --steps 16384 --seed 7 \
  --resume runs/exploration_stage1/hold_s7_pilot/final.pt \
  --output runs/exploration_stage1/reach3_s7_pilot

bash scripts/run.sh -m wheel_manipulation.exploration.evaluate \
  --control zero --output outputs/exploration_stage1/zero_suite

MUJOCO_GL=osmesa LP_NUM_THREADS=1 bash scripts/run.sh \
  -m wheel_manipulation.exploration.evaluate \
  --model runs/exploration_stage1/reach3_s7_pilot/final.pt \
  --video-case sequence --output outputs/exploration_stage1/reach3_suite
```

训练产物包括 `config.json`、`training.csv`、`episodes.csv`、checkpoint、`final.pt` 和 `actor.pt`。Checkpoint 包含新接口版本及优化器；不能加载旧 18 维 PPO 的权重来假装续训。

导出 actor 的调用为 `action, next_hidden = actor(observation, hidden, episode_start)`，张量分别为 `[batch,140]`、`[batch,128]`、`[batch]`。导出图不含 critic 或特权输入。

完整的匹配试验及静态汇总可用下列命令复现；评价输出目录必须不存在或为空，各组严格顺序运行。名义五场景仅用一个确定性准备种子，随机化序列使用三个准备种子，每个控制器共八回合。

```bash
MUJOCO_GL=osmesa LP_NUM_THREADS=1 bash scripts/run.sh \
  scripts/evaluate_exploration_stage1.py --output outputs/exploration_stage1_recheck
bash scripts/run.sh scripts/summarize_exploration_stage1.py \
  --input outputs/exploration_stage1_recheck --output results/exploration_stage1_recheck
bash scripts/run.sh -m unittest discover -s tests -v
```

离线测试对照零策略、保持训练模型和小角度训练模型，使用相同目标/准备种子。重点看保持漂移、正负目标误差、反向误差、完整执行率、接触与异常载荷，并保留逐步 CSV。保持模型未训练过非零目标，这些场景只是冻结模型诊断。

探索精度门槛：保持整段及末 5 秒 RMSE 均不超过 1°；正负阶跃末 5 秒 RMSE 不超过 1°；正弦及往返整段 RMSE 不超过 1°、峰值误差不超过 5°；所有场景须完整执行。`max_hold_seconds` 仍是原有 2°误差/速度阈值的粗诊断，不能单独作为精度验收。最初两个 `*_hold_smoke` 记录采用较宽的旧诊断门槛；正式三组对照以修订后的评价器和 `criteria` 字段为准。

验收标记始终保持 false，直到有足够独立长回合和多训练种子证据。完成回合只代表未触发当前仿真终止条件，不代表实物安全。

结果记录在 [第一阶段结果](../results/exploration_stage1/README.md)。
