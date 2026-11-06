> 2026-10-05：代码已独立到本项目`src/wheel_manipulation/`；本页命令在`manipulation/`内运行。

# 共同圆弧运动与臂手接触修正

这是独立操作任务，不连接 XY 路径规划、视觉或 CARLA。初始握持、轮轴位姿和轮辋几何已知；轮盘保持被动，只由手轮接触驱动。接口版本为 `dexsteer-v2-contact-residual`，上一版 18 维速度动作权重不能用于这里。

## 控制边界

仿真内生成目标轮角及角速度 → 确定性的轮角反馈/前馈 → 双腕共同圆弧参考 → PPO 的有界手指/腕部修正 → 连续微分 IK/QP → 有限刚度 PD/力矩限幅 → 实际轮角。

共同参考速度为 `target_velocity + 2 * angle_error - 0.15 * velocity_error`，限速 20°/s、参考加速度 80°/s²。实际轮速由 50 Hz 轮角差分及滤波估计。轮盘编码器与腕部参考相位分别记录；抓握滑移时它们可以不同。共同参考留有残差余量、工作空间制动距离和实测腕部偏差约束，饱和时不继续积累参考；约束不可行终止回合。

参考圆弧来自初始掌部坐标和已知轮轴；掌部参考半径与物理轮辋半径不同。两臂关节不按稀疏 IK 表插值，QP 连续求解关节速度，并限制参考位置、速度和加速度。实际关节速度、接触负载和参考跟踪误差需要单独测量；参考限幅不是实物安全证明。

独立场景 `assets/wheel.xml` 使用 MuJoCo affine PD actuator，让 `implicitfast` 对阻尼项进行隐式积分。刚度、阻尼、重力/偏置前馈和总电机力矩上限沿用原参数；没有添加无限制外力或被动电机阻尼。从相同预抓握物理状态开始，以500Hz记录1.2秒初始化和10秒中心保持，显式执行的实际最大臂速为6.674rad/s、右腕相邻速度相关系数为−0.997；隐式执行对应为0.336rad/s和+0.990。保持阶段轮角RMSE由0.502°变为0.291°，最终握持漂移代理由6.687°变为0.581°。两组总电机力都没有超出限幅。若先从已稳定的隐式抓握状态开始，显式执行也可稳定，因此这项结果说明初始化相关的数值问题，不能断言所有显式PD都不稳定。原始500Hz配对见 `outputs/validation_physics500hz/`；修正前的试训与场景保留，不能作为本场景权重恢复。

## 策略接口

20 Hz 非对称 PPO，50 Hz 臂手参考更新，500 Hz 物理。Actor `[256,256,128]`、critic `[512,512,256]`、ELU。初始 actor 输出为零修正；主要转动控制器明确位于执行层。

| 动作索引 | 含义 | 初始范围 |
|---|---|---|
| 0–11 | 左右手六驱动位置修正，围绕已知闭手目标 | ±0.15 rad，再限于驱动/联动允许范围 |
| 12 | 共同绕轴相位修正 | ±1° |
| 13 | 两腕差分相位修正 | ±0.5° |
| 14–15 | 左腕径向/轴向位置修正 | 各 ±2 mm |
| 16–17 | 右腕径向/轴向位置修正 | 各 ±2 mm |

左右腕相位分别为 `common_reference + common_residual ± differential_residual`，最大相位差 1°。这些是位置修正目标，经时间常数 0.1 s 的滤波及变化率限制；恒定输出不会独立积分成越来越大的位移。

每个 actor 帧 122 维：关节位置26、估计关节速度26、电机目标与位置之差26、上一动作18、轮轴坐标中的双腕位置/旋转18、轮角/轮速/目标角/目标速度/误差/轮辋半径6、共同参考相位/速度2。历史八帧合计976维。Actor 不读取模拟接触力。Critic 额外读取54维特权信息，部署导出的 `actor.pt` 只接收 `[batch,8,122]`。

奖励优先考虑实际轮角误差、速度误差和握持漂移，并辅以局部轮辋包络和接触；惩罚模拟负载峰值、碰撞及剧烈/过大修正。包络和漂移是几何代理，不是触觉或力闭合测量。

## 实验与复现

固定手指、仅手指RL、手指加腕部RL使用同一场景、共同控制器、目标序列、随机化与评价协议。仅手指对照单独训练12维动作策略；不能简单屏蔽已训练18维策略后宣称这是同一对照。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DRIVE_PYTHON=/home/nubot/anaconda3/envs/mujoco/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/humanlike-drive-mpl

# 固定握持、共同控制器；不是 RL 演示。
"$DRIVE_PYTHON" -m wheel_manipulation.evaluate \
  --control nominal --pilot --seconds 60 --output outputs/fixed_grasp_new

# 单环境握持试训；所有训练共用进程锁，不能同时启动。
"$DRIVE_PYTHON" -m wheel_manipulation.train \
  --stage grasp --steps 32768 --seed 7 --output runs/grasp_new

# 独立保持/小角度评估。
"$DRIVE_PYTHON" -m wheel_manipulation.evaluate \
  --model runs/grasp_new/best/best_model.zip \
  --pilot --seconds 60 --output outputs/grasp_new_eval

# 通过握持验证后，再续训小角度；写到新目录。
"$DRIVE_PYTHON" -m wheel_manipulation.train \
  --stage reach10 --steps 200000 --seed 7 \
  --resume runs/grasp_new/best/best_model.zip \
  --output runs/reach10_new

# 仅手指对照：独立12维模型，共同控制器不变。
"$DRIVE_PYTHON" -m wheel_manipulation.train \
  --hands-only --stage grasp --steps 32768 --seed 7 --output runs/hands_new
```

课程为稳定握持 → ±10°到达/保持 → ±30° → 动态跟踪 → 渐进随机化。各阶段先小规模试训，独立验证不过不晋级。保持目标：误差≤2°、轮速≤2°/s、双手有效握持并连续10秒；动态目标：600秒RMSE≤2°、最大误差≤5°。目标尚未验收，不能用低RMSE代替停止保持，也不能将共同控制器的效果算成RL贡献。

权重、原始CSV、视频在本机 `runs/`、`outputs/`。训练记录接口、动作含义、配置、场景/依赖源码哈希、步数和时间；所有checkpoint必须通过v2元数据校验。

课程脚本支持按阶段验证后晋级、按种子顺序训练；全套最终评价使用与开发试验不同的8001–8005测试种子。下面的命令是后续完整验收入口，尚未执行：

```bash
"$DRIVE_PYTHON" -m wheel_manipulation.curriculum \
  --seeds 7 17 27 --steps-per-stage 200000 --output runs/curriculum_new
"$DRIVE_PYTHON" -m wheel_manipulation.evaluate \
  --suite --stage robust --randomization 1 \
  --model runs/curriculum_new/s7/robust/best/best_model.zip \
  --output outputs/final_s7_new
```

`--suite`评价五个保持角及慢/快正弦、停留反向、混合轨迹；动态回合各600秒。单个checkpoint过关仍不等于三个独立训练种子的最终验收。

## 当前初步结果

主18维策略已经完成握持与±10°到达阶段，各新增32768步。第二阶段从握持阶段8192步的最佳checkpoint恢复，所以最终模型累计40960步；训练计算预算合计65536步，二者不能混用。仅手指12维策略是单独训练的对照，主方法仍包含腕部修正。

主策略在−10°/0°/+10°三项60秒测试中均通过连续10秒保持。±10°、0.02Hz、600秒的轮角RMSE为0.06293°，固定握持对照为0.06465°；低误差主要由共同反馈提供，尚不能说明RL显著改善跟踪。在一次120秒随机化样本中，主策略将握持漂移代理峰值从4.629°降至2.989°，但单接触力峰值从3.751N增至4.508N。120秒停留反向的最大误差仍约6.38°，未达到5°上限。

完整三组配对、训练谱系、执行层诊断与限制见 [初步实验报告](../results/pilot/README.md)。[30秒演示视频](../outputs/implicit_full_sequence_demo/rollout.mp4)使用确定性共同运动加学习的接触修正，不包含上游驾驶。当前只验证小角度开发样本；尚未完成±30°、多训练种子、换手或实物迁移。

## 限制

目前是仿真及已知握持起点，没有重新抓握、上游驾驶或实物发布器。手指联动、摩擦、接触顺应和执行参数未实物标定；真实轮角和驱动单位也待验证。当前模型中的局部自碰撞过滤沿用独立操作场景。参考控制、接触模拟及500Hz数值执行均需分别检查，不能把不合理的仿真抖动当作策略应该学习的能力。
