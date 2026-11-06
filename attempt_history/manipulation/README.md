# 双臂双手方向盘 manipulation

本目录是独立Python项目，包名`wheel_manipulation`。复制本目录并提供Python依赖、机器人源模型后即可运行，不需要baseline目录。

任务：已知轮轴和轮辋几何、已知初始抓握，上身固定，双手通过物理接触转动被动方向盘；不接CARLA、视觉或路径规划。

2026-10-23按用户要求新增[双手简化任务](docs/HAND_ONLY.md)：物理执行移除机械臂，两只浮动Inspire手通过有限力/力矩的虚拟腕部执行器运动，保留六驱动、mimic联动与物理接触。现已加入独立H1运动学可达性守卫，接口为137维观测、24维动作；已完成[可达性与物理诊断](results/hand_only/reachability_v1/README.md)，尚未为新接口训练策略。目标与后续课程见[双手中心设计](docs/HAND_CENTRIC_DESIGN.md)。原有臂手任务与模型保留，以下为其记录。

当前换手研究入口：目标角/角速度与预览 → 集中式腕手策略 → 柱坐标腕运动/手指目标 → 联合IK/QP → 限幅PD → 接触 → 被动方向盘。原有固定抓握残差实验保留独立入口和产物。

## 文件布局

| 目录 | 用途 |
|---|---|
| `src/wheel_manipulation/` | 场景、环境、控制、PPO训练、评价与课程代码 |
| `src/wheel_manipulation/common/` | 本项目内部的机器人参数、接触/渲染、场景模板、IK、网络和验收工具 |
| `assets/`、`configs/` | 当前`wheel.xml`场景和`task.yaml`配置 |
| `scripts/`、`tests/` | 统一运行入口、诊断、结果汇总和验证 |
| `runs/` | checkpoint、actor、训练元数据与日志 |
| `outputs/` | 原始CSV、视频、接触检查及执行层诊断 |
| `results/`、`docs/` | 轻量结果、迁移验证与技术文档 |
| [`../references/`](../references/README.md) | 根目录统一论文库、阅读索引和专题笔记 |

## 当前状态

第一阶段换手方案正在执行，代码在 [`regrasp/`](src/wheel_manipulation/regrasp/)，设计见 [CONTACT_REGRASP_DESIGN.md](docs/CONTACT_REGRASP_DESIGN.md)，实际记录见 [阶段工作记录](results/regrasp_stage1/README.md)。这是 152 维部署观测、54 维训练特权输入、24 维臂手联合动作的教师/历史学生路线。当前在训练和诊断教师，尚未证明完整换手，也未开始合格教师的学生蒸馏。

该执行层没有轮角误差伺服；柱坐标是几何运动表示，双腕运动仍由策略给出。后续课程会逐步覆盖单手控制、接管与重抓、连续反向；小角度结果不会作为整个第一阶段通过的依据。模型和原始评价分别写入 `runs/regrasp_stage1/`、`outputs/regrasp_stage1/`，与以下旧实验分开。

2026-10-22 新增独立的[集中式腕手 GRU 第一阶段探索](docs/EXPLORATION_STAGE1.md)：140 维单步观测、24 维动作，策略自主生成双腕运动，保留 IK/QP 执行；与下述已有 18 维残差模型分别保存。[探索结果](results/exploration_stage1/README.md)。

首轮累计训练 24,576 步，完成三组共 24 个 15 秒对照回合；新策略尚未通过 ±3°精度门槛，存在共同正向偏移，不能作为已可用的驾驶/换手策略。下述低误差结果属于原有圆弧参考加残差模型。

Actor输入8帧×122维，包含本体状态、轮角、当前目标及本地参考状态；不读模拟接触力。18维主动作是12个手指修正加6个有界腕部修正。独立12维仅手指模型是消融。

已完成seed7的握持和±10°试训，以及保持、正弦、停留反向和一次随机化评价。慢正弦600秒RMSE约0.063°，主要来自共同反馈；停留起步峰值约6.38°。±30°、多训练种子、换手和实物迁移尚未完成。

- [完整pilot报告](results/pilot/README.md)。
- [技术路线和接口](docs/METHOD.md)、[与基线的差异](docs/COMPARISON.md)。
- 主模型：`runs/implicit_reach10_s7_pilot/best/best_model.zip`，同目录`actor.pt`输入`[batch,8,122]`。
- 消融：`runs/implicit_hands_reach10_s7_pilot/best/`。
- [演示视频](outputs/implicit_full_sequence_demo/rollout.mp4)、[初始接触诊断](outputs/initial_grasp_contacts/README.md)。
- `runs/grasp_s7_pilot`是本方案早期显式执行的诊断，不是已删除的速度PPO版本，也不是当前主模型。

## 运行

从本目录执行。`scripts/run.sh`设置`PYTHONPATH=src`并使用现有`/home/nubot/anaconda3/envs/mujoco/bin/python`；也可通过`DRIVE_PYTHON`指定解释器。使用新输出目录。

```bash
bash scripts/run.sh -m wheel_manipulation.evaluate \
  --model runs/implicit_reach10_s7_pilot/best/best_model.zip \
  --stage reach10 --pilot --seconds 60 --output outputs/pilot_recheck_new

bash scripts/run.sh -m wheel_manipulation.evaluate \
  --control nominal --pilot --seconds 60 --output outputs/nominal_recheck_new

bash scripts/run.sh -m wheel_manipulation.train \
  --stage grasp --steps 32768 --seed 7 --output runs/grasp_new

bash scripts/run.sh scripts/summarize_results.py
bash scripts/run.sh -m unittest discover -s tests -v
```

旧残差入口使用单环境；`regrasp.train` 默认在一个训练作业内使用4个CPU环境。训练共用进程锁，须顺序执行，评价也应在训练结束后执行。渲染可加`MUJOCO_GL=osmesa LP_NUM_THREADS=1`。默认源模型在`/home/nubot/unitree/unitree_ros/robots/h1_2_description/`，可用`H1_MODEL_PATH`指定。

## 历史模型兼容

旧checkpoint字节未改写；`wheel_manipulation.common.policy.load_policy`将历史SB3策略类引用映射到本地同一结构，训练及评价入口均使用它。原始JSON中的旧路径和源码哈希保留历史含义。接口标识`dexsteer-v2-contact-residual`继续用于校验模型；不是保留旧速度任务。

目录迁移记录见[isolation.json](results/isolation.json)，此前迁移见[reorganization.json](results/reorganization.json)。
