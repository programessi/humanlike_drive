# 第一层：掌心驱动方向盘的非学习工程系统

本系统回答“在已声明的仿真条件下，是否存在一套真正依靠接触完成任务的操作过程”。实际通过的范围、数值与失败案例统一见[层1验收报告](../results/physical_feasibility/l1_v1/README.md)。这份说明介绍如何运行及各部分承担什么职责；它不是训练方案。

## 1. 操作过程

从坐姿驾驶室中的左9点、右3点建立掌面接触。手掌向盘心施加径向预紧，再沿轮圈移动，通过切向摩擦转动轮盘。四指保持统一轻握目标，依靠现有有限位置驱动产生顺应；不增加侧摆自由度，也不要求每根手指接触。手指的辅助作用是否成立，要看消融与实际受力，不能从手型外观推断。

整个系统由五个相连的部分组成：

1. **路径库**：离线IK求出不同空间相位的自然腕姿和退让姿态，检查躯干、外壳等间隙，再做连续插值。离线数据只改求解器的临时状态。
2. **角度参考与接力安排**：把累计轮角目标序列分成短行程。每只手主要在自己一侧的上半圆工作，到窗口边缘就停稳接管。累计轮角不取模，换手不重置轮角或控制器历史。
3. **掌压和轮角反馈**：依据掌面法向力微调径向压入量；依据轮角误差微调切向目标。接管期间，新手还根据独立承担的轮轴转矩调整小量切向偏置。
4. **执行器**：将目标映射到原来的26个驱动，在20 Hz限制目标变化，在500 Hz插值并执行有限位置电机。重力／模型偏置补偿与电机输出之和仍受原输出上限约束。轮轴没有伺服。
5. **独立验收**：逐物理步保存轮角、命令、输出、接触与滑移，分别检查任务、干涉、执行约束、掌面摩擦传力和接管，而不是只看视频。

当前选择较短行程，是为了避免两手在12点附近挤在一起，并给接近与退让保留余量。这不是“每只手必须转180°”的动作模板；后续可以在相同接口上比较更长窗口或更高效的换手规划。

## 2. 怎样接管

顺序为：旧手保持 → 新手退让状态下回到准备相位 → 接近 → 掌压稳定 → 转移负载 → 旧手退出 → 新手独立保持10秒 → 新手继续转动。

转移前，最近0.2秒内新手掌面接触力超过0.5 N的占比至少95%。释放前，需观测到至少连续0.1秒的新手承担至少一半反负载转矩，且双掌确实接触。进入独立保持前，旧手轮圈总力必须降到0.05 N以下。接触或转矩条件尚未满足时，允许额外最多5秒保持上一稳定状态，旧手继续承载；超时或旧手未能清空时停止并保存失败原因。

这是一套显式状态机加接触反馈的控制器，使用仿真中的掌力和逐手接触力矩真值。实物需要对应传感或估计能力；不能把这些真值直接当成未来学生策略的部署观测。

## 3. 模型与数值边界

保持原坐姿驾驶室、G29视觉模型、拨片／轮辐／底座、原关节限位及URDF手指mimic。只排除文档化的相邻装配体重叠，清单随运行保存。

原掌部整体凸包会填平CAD凹处，产生距离视觉表面数毫米的假接触。新碰撞体将原始掌部三角面按5 mm列裁切后分别取凸包，不改变视觉、惯性和自由度。列宽不是误差上界：每条代表轨迹另抽样检查受载接触点到原始掌部及轮圈视觉CAD的距离，要求≤0.5 mm。

摩擦锥、阻抗比、求解精度等数值参数写在控制配置的`simulation_solver`中，并落到每次运行的XML。接触软约束可能产生慢滑移，必须报告数值敏感性；不能把调过求解器的结果与旧设置混为同条件结果。数值改进不等于接触材料已经实物标定。

负载采用明确的恒定轮轴转矩叠加黏性阻尼，并保留原关节阻尼、干摩擦、重力和转子惯量。它是合成负载，尚非实测G29力反馈曲线。原装模型居中±450°，不能用于验证居中向任一侧540°。

## 4. 运行入口

在项目根目录执行。以下目录必须全新，避免覆盖证据。

```bash
# 1. 用有限电机建立初态；不固定轮轴
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.physical_feasibility prepare \
  --output outputs/my_l1/preparation

# 2. 核查初态受载接触是否贴合原CAD
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.l1_verify cad \
  --run outputs/my_l1/preparation --samples 100

# 3. 几何筛查两手沿上半圆的可达与干涉
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.physical_feasibility scan \
  --grasp outputs/my_l1/preparation --output outputs/my_l1/workspace

# 4. 累计轮角180°、保持、回正；镜像测试把180改成-180
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.physical_feasibility run \
  --grasp outputs/my_l1/preparation --case relay --angle 180 \
  --output outputs/my_l1/positive180

# 5. 自定义连续目标序列；中途不重置
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.physical_feasibility run \
  --grasp outputs/my_l1/preparation --case relay \
  --goals-deg 60 30 -30 0 --output outputs/my_l1/mixed

# 6. 本机窗口回放，空格播放，V切相机，鼠标自由观察，+/-变速
bash scripts/run.sh -m humanlike_drive.env_runner.replay_p0 \
  --run outputs/my_l1/positive180

# 7. 无窗口导出完整视频；speed=1为原速，4为四倍速
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.physical_feasibility render \
  --output outputs/my_l1/positive180 --camera driver --fps 15 --speed 1

# 8. 从首帧重新执行已记录命令，验证日志可重现真实物理
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.l1_verify replay \
  --run outputs/my_l1/positive180 --seconds 20
```

`roundtrip`用于局部单手往返，`hold`用于逐手承载，`relay`用于带接力的累计轮角目标，`crosszero`用于固定的途中反向与跨零序列。`--duration`只用于调试，截断轨迹不会计作完整通过。可用`--load`、`--friction`、`--target-offset-mm`和`--seed`预先声明负载、摩擦及掌面目标标定偏差，配置与初态不符时要求重新准备。

`--target-offset-mm`通过有限电机引入目标配准偏差，保留合法的实际初态。旧`--offset-mm`直接扰动仿真初态，可能造成初始穿透，只作为开发探针，不用于本轮合法初态鲁棒性结论。

完整矩阵入口：

```bash
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m humanlike_drive.env_runner.l1_suite \
  --grasp outputs/my_l1/preparation \
  --config humanlike_drive/config/controller/palm_primary_l1.json \
  --output outputs/my_l1/suite --jobs 2
```

`--groups nominal load robust diagnostic`可选择组别，`--only`可选择明确案例名；`extension`另测360°。大负载与扰动是探测范围，失败照实保留，不等于验收通过。名义组和负对照是否符合预期分别报告。

## 5. 证据在哪里

| 文件 | 含义 |
| --- | --- |
| `scene.xml`, `config.json`, `manifest.json`, `source_snapshot/` | 实际场景、参数、软件版本、源文件及外部CAD哈希 |
| `grasp.json`／`grasp_used.json` | 准备后实际初态及任务使用的初态 |
| `trajectory.npz` | 500 Hz状态、参考、实际26维电机控制、20 Hz动作／执行器内部目标、补偿和接触统计 |
| `contacts.npz`, `contact_schema.json` | 原始接触几何对、位置、坐标系、六维力／力矩、掌面局部坐标及相对滑移 |
| `schedule_nominal.json`, `schedule.json`, `transition_guards.json` | 原始时间计划、考虑有界等待后的时间安排和实际转换检查 |
| `evaluation.json`, `collision_events.json` | 验收判定、指标和碰撞事件 |
| `contact_CAD_audit.json`, `dynamics_replay_check.json` | 独立CAD抽样与命令重执行检查 |
| `workspace.json`, `workspace_poses.npz`, `workspace.png` | 几何筛查结果；它们不是动态成功证据 |
| `tracking.png`, `replay_*.mp4` | 轮角／受力诊断与完整状态回放 |

接触力转世界坐标后，按真实轮轴计算`轴向力矩 = 轴方向 · (力臂 × 力 + 接触力偶)`。掌、拇指、四指、其他表面分别统计，法向与切向摩擦分别统计。再与广义接触约束力矩、包含转子惯量和干摩擦的轮轴动力学平衡核对。驱动、制动、保持与两种力矩方向分别汇总，不用正反抵消后的净值证明掌心主驱。

`l1_verify replay --interface actions`还会将记录的20 Hz归一化动作重新送入执行器，重建500 Hz插值，并重新计算有界补偿和负载，用于核实后续专家动作接口。

窗口／视频通过写入记录状态实现可视化，不产生新的物理证据。`l1_verify replay`才是从首状态依次重执行命令；两者明确区分。

## 6. 怎样接到第二层

通过验收的完整轨迹可作为**仿真工程专家候选**。它们包含实际执行动作和控制状态，可以用于确认动作接口、采样频率和目标定义。当前没有人类动作标签、视觉训练集或已训练策略，不能直接称作人类示范或成熟BC数据集。

下一步在已证明物理可行的同一范围内适配已有学习方法，保持碰撞、负载和有限输出一致。学习失败时先检查观测、动作接口、示范覆盖和训练预算，再讨论方法创新。真实实验仍先执行[R0标定](plans/R0_HARDWARE_CALIBRATION.md)。
