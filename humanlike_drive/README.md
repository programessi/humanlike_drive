# Humanlike Drive：新一轮方向盘操作研究

本目录是新尝试的独立项目根目录：`/home/nubot/humanlike_drive/humanlike_drive`。其中同名的 `humanlike_drive/` 是 Python 源码包，不是另一套实验。创建日期：2026-11-05。

当前已实现一套重新构建的 **H1-2 / Inspire 双手 / Logitech G29 坐姿驾驶室**，包含大屏幕、仪表台、座椅、中控、踏板，以及四种基础抓握预设。场景不加载旧场景、旧抓握数据或旧策略；策略训练和实物控制尚未开始。

## 当前研究主线

[研究计划v4.2](docs/plans/RESEARCH_PLAN.md)按三层推进：

1. **物理可行性：**显式控制／规划验证局部接触、接管和代表性完整任务，给出设备与负载范围。
2. **现有方法适用性：**筛选相近工作与可复用开源实现，合理适配少量代表性方法，测出能力、成本和失败边界。
3. **针对性提升：**有明确、可重复的不足后再做原因诊断与方法设计；不预设网络或创新模块。

场景、查看器、原始结果和代码保留；[历史尝试凝练为经验](docs/experiments/HISTORICAL_LESSONS.md)。此前左右90°已[独立备份](results/archive/20261107_pre_palm_primary/README.md)，仅作历史参考，不自动成为新主线数据或通过证据。

当前第一层采用**掌心径向预紧、切向摩擦主驱，手指顺应维持**；优先从9点／3点自然腕姿出发，检查两手上半圆可达性与干涉，再逐级向0—180°受载往返推进。360°／540°轨迹有条件则扩展和保留为专家数据候选，不作为当前阶段前置要求。[第一层物理协议](docs/experiments/PALM_PRIMARY_P0.md)与[第二层评价协议](docs/experiments/EXISTING_METHODS_EVALUATION_V1.md)规定证据要求；[B0草案](docs/plans/BASELINE_IMPLEMENTATION_V1.md)仅作为第二层候选。层1非学习工程系统已实现，实际验收范围见[新层1报告](results/physical_feasibility/l1_v1/README.md)，使用方法见[工程系统手册](docs/LAYER1_ENGINEERING.md)。学习训练尚未开展，实物仍按[R0](docs/plans/R0_HARDWARE_CALIBRATION.md)后续落实。

## 当前层1入口

掌心摩擦主驱的新控制器使用离线自然腕姿路径、轮角／掌压反馈和带真实负载检查的上半圆接力，原26维有限驱动与被动轮轴保持有效。当前名义速度为1°/s，采用未标定的0.1 N·m合成负载。它用于验证物理可行性，不是学习策略或实物验收。

- [层1报告、完整视频与通过边界](results/physical_feasibility/l1_v1/README.md)。
- [系统组成、一键复现、查看及动作重执行](docs/LAYER1_ENGINEERING.md)。
- [冻结控制配置](humanlike_drive/config/controller/palm_primary_l1.json)。

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.replay_p0 \
  --run outputs/physical_feasibility/l1_v3_20261107/core/positive180
```

## 历史仿真P0结果与回放

[本轮仿真P0报告](results/p0_complete_20261107/README.md)汇总双向30°／60°／90°往返、左右单手承载、静止和低速交接，以及含同侧反向和跨零的连续序列；报告保留向±180°扩展时的失败边界，不能据此宣称完整±180°或多圈策略已完成。实物接口与标定按用户要求在[R0](docs/plans/R0_HARDWARE_CALIBRATION.md)后续单列。

查看完整连续序列（默认暂停，空格播放，V切换相机，鼠标调整视角，+／-变速）：

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.replay_p0 \
  --run outputs/p0_complete_20261107/continuous_negative70_v5
```

重跑旧协议回归需使用新的输出目录；下面命令属于历史路线，不是新P0入口：

```bash
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh -m humanlike_drive.env_runner.p0_suite \
  --output outputs/p0_regression_new --jobs 2
```

## 先看场景

在本项目根目录执行：

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene
```

默认暂停。`1–4` 切换双手9/3、双手10/2、左手接触、右手接触；`V` 切换相机；`F` 进入可用鼠标操作的自由视角；`C` 查看碰撞体；`T` 半透明显示实际受力接触点；空格运行／暂停物理；`R` 重置当前抓握。

抓握已按反馈修正无名指、小指误压后方拨片的问题，接触统计明确区分轮圈与拨片／轮辐；mimic仍从同一套H1-2 URDF导入。[第3版抓握报告](results/grasp_refinement_v3/README.md)包含侧后方对照与5秒保持检查。第2版关于“四指支撑轮圈”的结论已作废。

当前第5版参考真实驾驶员照片与驾驶教学视频抽帧，联合调整1号手掌朝向、肩肘位置和六驱动闭合量，将腕部内扣从约35°降到5.3°，前臂与手掌夹角约9°。四指、掌部和拇指持续支撑轮圈，未压到拨片；2–4号保留原预设。见[本次调整及参考来源](results/driverlike_grasp/README.md)、[同视角前后对照](results/driverlike_grasp/before_after_driver.png)。

- [驾驶室预览](results/logitech_scene_v1/nine_three_cabin.png)、[完整图集](results/logitech_scene_v1/index.html)、[抓握预览](results/logitech_scene_v1/grasps_overview.png)。
- [场景说明与重建方法](docs/SCENE_LOGITECH_G29.md)：模型来源、参数、抓握准备过程和验证范围。
- [三层研究计划v4.2](docs/plans/RESEARCH_PLAN.md)：阶段问题、完整可行性、已有方法适配、针对性提升与实物验证。原静态抓握预设保留作查看工具；当前掌心主驱初态与结果以新层1报告为准。

脚本默认使用本机已有的 `mujoco` Conda 环境，也可通过 `HUMANLIKE_PYTHON=/path/to/python` 指定解释器。另建环境时可安装 `pip install -e '.[scene]'`。

## 组织依据

参考 [Diffusion Policy 官方仓库](https://github.com/real-stanford/diffusion_policy) 将配置、数据集接口、网络、策略、实验工作流、环境与评估执行器分开的组织方式。这里只借鉴目录职责，没有复制其实现，也不表示已经选择扩散策略。布局参考于2026-11-05核对。

## 目录

```text
humanlike_drive/                   本项目根目录
├── README.md
├── pyproject.toml                 包元数据与scene可选依赖；算法依赖以后补充
├── humanlike_drive/               Python源码包
│   ├── config/                    可组合配置；后续用于Hydra等配置工具
│   │   ├── task/                  任务范围、参考、奖励和终止配置
│   │   ├── policy/                策略和记忆、动作序列配置
│   │   ├── model/                 神经网络结构配置
│   │   ├── dataset/               数据路径、采样和归一化配置
│   │   ├── workspace/             RL、BC、蒸馏等实验工作流配置
│   │   ├── env_runner/            开发、回归、独立评估配置
│   │   ├── controller/            IK/QP、动作尺度与执行限制配置
│   │   └── real_world/            设备、反馈、延迟和标定配置
│   ├── common/                    共享工具、几何、序列缓冲和检查点工具
│   ├── task/                      任务定义、参考轨迹、观测/动作约定、奖励与事件
│   ├── model/                     神经网络组件：编码器、循环/Transformer模块、输出头
│   ├── policy/                    统一腕手策略：历史、记忆、动作预测和推理接口
│   ├── dataset/                   数据读取、序列采样、转换和数据校验代码
│   ├── env/                       仿真环境、reset、step及物理交互
│   ├── env_runner/                驱动策略评估，汇总逐例结果、指标和视频
│   ├── controller/                腕手动作执行、IK/QP、限速与有限输出
│   ├── workspace/                 完整训练工作流、恢复、日志和周期评估
│   └── real_world/                真实设备、同步采集、遥操作及部署接口
├── assets/
│   ├── robots/                    机器人描述、资源索引及必要模型文件
│   ├── objects/                   方向盘等物体模型
│   └── scenes/                    仿真场景与组合文件
├── data/
│   ├── raw/
│   │   ├── human/                 人类动作/交互示范原始数据
│   │   ├── robot/                 实物机器人示范、反馈和人工干预记录
│   │   └── sim/                   仿真RL轨迹、专家示范及纠正数据
│   ├── processed/                处理后的统一序列数据；存储格式待定
│   ├── manifests/                数据来源、版本、字段和校验和清单
│   ├── splits/                   训练/验证/开发/独立留出划分
│   └── calibration/              设备及模型标定记录
├── scripts/                      run.sh及后续数据处理、采集和实验入口
├── tests/                        后续接口与序列检查；当前物理检查在env_runner/audit_scene.py
├── docs/
│   ├── plans/                    研究与实施计划
│   ├── decisions/                经讨论确认的方案选择与依据
│   └── experiments/              每项实验的预声明协议、预算和验收条件
├── outputs/                      每次运行的配置、日志、权重、轨迹和视频
└── results/                      精选结论、验收报告、图表和正式模型索引
```

## 关键职责

- `model/` 是可复用网络组件；`policy/` 组合这些组件并定义观测到动作的行为。
- `workspace/` 负责训练与恢复；`env_runner/` 负责冻结策略的闭环评估。
- `task/` 定义任务及接口；`env/` 实现物理环境；`controller/` 把策略动作落实为执行命令。
- `dataset/` 放代码，`data/` 放数据。`assets/` 放模型资源，`outputs/` 放实际运行产物。
- 后续若沿用Diffusion Policy入口习惯，可在项目根增加薄入口 `train.py`、`eval.py`；具体逻辑放在源码包中。本轮未创建空的可执行入口。

## 数据与实验约定

1. 原始数据按来源归档，保留原始时间戳；数据转换生成新版本，不覆盖原始记录。
2. 每份可训练数据由manifest说明来源、观测/动作版本、单位、频率、有效片段和校验和。失败和人工干预要有标记。
3. 划分按轨迹、采集会话或种子等分组生成，避免把同一轨迹相邻帧分到训练与验证。
4. 每次实验新建 `outputs/YYYY.MM.DD/HH.MM.SS_<name>/`，保存完整解析配置、代码版本、随机种子、数据清单、日志、检查点和评估结果。
5. 正式结论写入 `results/`，链接对应运行及模型哈希。数据与训练、开发、独立验收的使用历史需可追溯。
6. 大体积原始/处理数据与运行输出默认不纳入Git；来源清单、划分、标定、协议和结果说明可版本管理。

## 与原项目的关系

父目录 `attempt_history/` 中的旧版本作为历史资料，统一论文库仍使用父目录 `references/`。这次从本机原始 H1-2 模型和 G29 URDF／CAD 重建场景，旧场景、抓握快照和训练数据不进入新环境。历史结果只能提供诊断经验；场景改变后不能直接作同条件性能比较。
