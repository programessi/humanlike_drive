# H1-2 驾驶室与物理方向盘控制

H1-2 固定坐姿，双臂和手指通过接触操作被动方向盘。MuJoCo 负责机器人与方向盘，CARLA 负责道路、车辆和前视图像。CARLA 的转向输入来自轮轴实际角度。

当前版本作为**基线**保存：视觉路径点 → 路径控制器 → 轮盘目标角 → 角度反馈/圆弧 IK + PPO 残差 → 关节 PD → 双手接触 → 实际轮盘角 → CARLA。机器人操作层是 residual RL；实验性的速度参考原型尚未替换正式控制。后续纯 RL 操作方法应与这个版本比较。

## 已实现

- 驾驶室场景：座椅、固定坐姿 H1-2、仪表台、方向盘、踏板、侧门、前方显示面；全景、手部近景和驾驶员视角。
- 14 个手臂关节和 12 个独立手部关节控制；其余手指关节由 12 条动力学 equality 约束联动。
- 分段轮圈碰撞、被动方向盘、逆运动学参考、限幅电机 PD 与角度反馈。
- Gymnasium 环境、PPO 残差策略训练、保存/恢复、评估指标、曲线及视频。
- ResNet-18 路径模型、CARLA 教师数据采集、视觉模仿学习和同步联合仿真代码。

本机已安装 CARLA 0.9.15；机器人 PPO 已训练累计 802816 步，最新视觉模型已用 9000 帧 Town04 连贯路线数据训练。已记录一次 300 秒、412.68 米、零碰撞、4 次压线的视觉机器人驾驶；前 200 秒轮盘 RMSE 为 0.60°，后 100 秒跟踪退化。该驾驶使用训练期间较早的视觉 checkpoint，不是最终第 12 轮权重。完整结果、模型哈希及 IK/握持问题见 [RESULTS.md](RESULTS.md)，轻量实验记录见 [baseline/](baseline/)。

Git 仓库保存源码、驾驶室配置、测试、文档和轻量基线记录。训练权重、数据集、视频、CARLA 安装包和第三方机器人网格保留在本机，不随 Git 推送；其他机器需要安装环境、提供 H1-2 模型并训练或复制权重后运行。

## 最快复现

```bash
cd /home/nubot/humanlike_drive
# 自动连接或启动本机 CARLA，生成驾驶视频和逐帧指标。
bash scripts/demo.sh --video

# 桌面窗口实时查看驾驶室，前屏显示真实 CARLA RGB。
# 需要本机图形桌面；--viewer 和 --video 分别运行。
bash scripts/demo.sh --viewer
```

演示脚本使用已训练权重，不重新下载或训练。输出目录为 `outputs/demo_日期_时间/`，包含 `drive.mp4`（视频模式）、`summary.json`、`drive_metrics.csv` 和带模型 SHA-256 的 `run_config.json`。可用 `--steps`、`--spawn`、`--output` 调整。

## 运行环境

本机已有环境，无需修改原机器人项目：

```bash
cd /home/nubot/humanlike_drive
export DRIVE_PYTHON=/home/nubot/anaconda3/envs/mujoco/bin/python
export MPLCONFIGDIR=/tmp/humanlike-drive-mpl
export PYTHONDONTWRITEBYTECODE=1
```

机器人原模型默认取自 `/home/nubot/unitree/unitree_ros/robots/h1_2_description/h1_2.xml`。其他机器可设置 `H1_MODEL_PATH` 后重新生成场景；模型同目录应包含 `meshes/`。项目依赖在 `pyproject.toml` 中，源模型和训练输出不纳入版本控制。

## 驾驶室预览与控制评估

```bash
# CPU 软件渲染导出三种视角。
MUJOCO_GL=osmesa "$DRIVE_PYTHON" -m humanlike_drive.preview --rebuild

# 桌面交互查看；有显示环境时使用。
"$DRIVE_PYTHON" -m humanlike_drive.preview --viewer

# 直接评估 IK+PD 基线。
"$DRIVE_PYTHON" -m humanlike_drive.evaluate --mode sine --amplitude 30 --seconds 30 --output outputs/baseline30

# 训练策略操作轮盘并输出视频、曲线和 CSV。
MUJOCO_GL=osmesa "$DRIVE_PYTHON" -m humanlike_drive.evaluate \
  --model runs/ppo_track/best/best_model.zip \
  --mode sequence --amplitude 15 --seconds 10 --video --output outputs/demo

# 关闭轮盘接触的因果对照；预期轮盘保持不动。
"$DRIVE_PYTHON" -m humanlike_drive.evaluate --no-contact --mode sine --output outputs/no_contact
```

## 操作策略如何训练

使用 Stable-Baselines3 PPO，actor 和 critic 分别是 `256 → 256 → 128` 的 ELU MLP。输入 112 维：关节角度/速度、参考姿态误差、上一动作、目标/实际转角、角速度和双手接触力。输出 26 维：14 个手臂关节参考修正、12 个手部关节参考修正。手臂残差默认 ±0.08 rad，手部残差默认 ±0.20 rad。

逆运动学给出轮圈弧线上的参考姿态，策略学习接触偏差和抓握修正。实际运动由限幅电机力矩与碰撞求解产生。轮盘没有电机，rollout 不写入轮盘 qpos；手部联动也不通过逐步改 qpos 实现。

奖励包含角度误差、角速度误差、手部接触、动作幅度/变化，以及过大接触力和非手部碰撞惩罚。建议依次训练：小角度随机目标保持 → 连续 ±30° 跟踪 → ±45° 与摩擦/阻尼随机化。步数是预算起点，是否进入下一阶段以独立评估为准。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$DRIVE_PYTHON" -m humanlike_drive.train \
  --stage hold --steps 2000000 --envs 4 --output runs/hold

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$DRIVE_PYTHON" -m humanlike_drive.train \
  --stage track --steps 4000000 --envs 4 --resume runs/hold/best/best_model.zip --output runs/track

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$DRIVE_PYTHON" -m humanlike_drive.train \
  --stage robust --steps 4000000 --envs 4 --resume runs/track/best/best_model.zip --output runs/robust
```

CPU 可训练，渲染默认关闭。并行环境使用 spawn。查看训练日志：`tensorboard --logdir runs`。恢复模型沿用 checkpoint 中的 PPO 参数；新任务配置保存在输出目录。

## CARLA 安装与启动

本机已安装官方 Ubuntu CARLA 0.9.15：服务器在 `vendor/carla-0.9.15`，匹配的 Python API 在 `vendor/carla-python`，不修改原 conda 环境。当前直接启动即可：

```bash
bash scripts/start_carla.sh
```

以下是重新安装时的命令（本机不需要重复下载/解压）：

```bash
python3 scripts/download_carla.py
mkdir -p vendor/carla-0.9.15
tar -xzf downloads/CARLA_0.9.15.tar.gz -C vendor/carla-0.9.15
bash scripts/start_carla.sh
```

另一个终端采集数据或联合运行：

```bash
# 初次接口测试：CARLA 道路真值产生目标，机器人实际转盘。
bash scripts/with_carla.sh -m humanlike_drive.carla_bridge \
  --mode teacher-robot --robot-policy runs/ppo_track/best/best_model.zip \
  --map Town04 --steps 1200 --speed 3 --video --output outputs/carla_teacher_robot

# 采集教师图像、车速和未来路径点。
bash scripts/with_carla.sh -m humanlike_drive.carla_bridge \
  --mode collect --map Town04 --steps 6000 --output data/town04
```

机器人控制 50 Hz，MuJoCo 积分 500 Hz，CARLA 同步推进 10 Hz；每次 CARLA tick 对齐相机 frame。退出后清理脚本创建的车辆与传感器，并恢复原 world settings。

桥接程序的 `--wheel-fullscale-deg` 默认 450，表示 900° 总映射；`--wheel-sign` 用于标定转向方向。当前固定握持目标限制约 ±54°，所以不能执行 G29 全范围操作。`--wheel-fullscale-deg 180` 明确表示 360° 总映射的模拟游戏设置，不是已经实现换手或实机 G29 标定。物理轮轴关节范围仍为 ±450°；只是角度到 CARLA steer 的映射不同。

`--target-rate-deg 20` 约束驾驶目标的变化速度，同时把目标角速度交给机器人策略。`--robot-guard` 在角度超过 nominal PPO 的训练区间或跟踪误差过大时，平滑减小 PPO 残差，回到 IK/PD。`--adaptive-speed` 根据视觉预测的转向强度和实测轮盘滞后降低速度。这些控制不读取地图真值。

## 视觉驾驶模型

ResNet-18 提取 `384×224` RGB 特征，融合车速和三维导航指令；`256 → 128 → 5` MLP 输出未来沿道路 3、6、9、12、15 米处路径点在车体坐标系中的横向位置。路径控制器产生转向目标，再交给机器人 PPO。推理使用固定前向距离近似这些点的纵向位置，急弯中此近似会有误差；正式扩展应预测完整二维路径。当前导航输入固定为道路跟随，路口路线选择和多指令训练尚未实现。

视觉层用 CARLA 道路教师做模仿学习：官方 ImageNet 权重初始化，2000 帧 pilot 训练 6 轮，再在完整数据上训练 12 轮；4800 帧训练、1200 帧时间块验证。当前 best 为完整训练第 12 轮，验证路径点 MAE 为 0.217 米。这是同地图的时间块验证，不能解释为跨地图/天气泛化。

```bash
"$DRIVE_PYTHON" -m humanlike_drive.vision --manifest data/town04/manifest.jsonl \
  --imagenet weights/resnet18-f37072fd.pth --epochs 30 --device cuda --output runs/vision

bash scripts/with_carla.sh -m humanlike_drive.carla_bridge \
  --mode vision-robot --vision-policy runs/vision/best.pt \
  --robot-policy runs/ppo_track/best/best_model.zip --steps 1200 --video --output outputs/visual_drive
```

视觉运行时地图只用于初始化与评估，控制目标由 RGB 模型产生。`lateral_error_m` 是相对于最近车道中心的带符号水平横向偏差；旧字段 `lane_error_m` 是到最近 waypoint 的三维距离，包含高度/采样误差，不能当成纯横向偏差。压线与碰撞事件也需一并查看，最近车道偏差本身不能证明车辆始终在同一车道。

导出实验曲线：

```bash
"$DRIVE_PYTHON" -m humanlike_drive.report outputs/carla_vision_robot
```

## 验证与当前边界

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$DRIVE_PYTHON" -m unittest discover -s tests -v
```

测试覆盖固定躯干/腿部、轮盘无执行器、无接触时轮盘静止、接触能够带动轮盘、Gym 接口、视觉模型反向传播与转向符号。

首版从预抓取姿态开始；完整接近抓取、换手、全转角操作和踏板控制尚未实现。座舱外观为程序化简化场景，轮圈尺寸按 G29 级别设置，摩擦和动力学参数未做实机标定。当前碰撞过滤关闭机器人内部自碰撞，需在实机迁移前完善；手部使用各指节凸网格，与真实柔性指腹也有差异。固定躯干和腿部是几何约束，不是学习平衡控制。
