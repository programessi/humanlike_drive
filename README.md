# H1-2 方向盘操作

根目录下保留三个独立项目。每个项目拥有自己的源码、包名、配置、场景、工具、模型和结果，运行时不导入另一项目。

```text
baseline/                      原CARLA驾驶基线
  src/drive_baseline/           基线源码
  assets/ configs/ scripts/ tests/
  runs/ outputs/ results/ docs/
  data/ weights/ third_party/   视觉数据、权重、CARLA
  pyproject.toml
manipulation/                  当前独立方向盘操作任务
  src/wheel_manipulation/       当前源码及本地基础模块
  assets/ configs/ scripts/ tests/
  runs/ outputs/ results/ docs/
  pyproject.toml
unified_steering/              新的统一臂手时间轨迹跟踪路线
  src/unified_steering/         已有审计、关节环境和PPO；腕手教师迁移待实施
  assets/ configs/ scripts/ docs/ results/ outputs/ runs/
  pyproject.toml
references/                    统一论文库、阅读索引和专项调研
```

- [论文阅读入口](references/README.md)：优先阅读列表、论文全文和各专题笔记。
- [统一臂手跟踪路线](unified_steering/README.md)：已抓握起步，采用结构化腕手动作、IK/QP及特权教师到历史学生；当前先复现旧教师并审计迁移差异。
- [最新研究与实施计划](unified_steering/docs/RESEARCH_PLAN.md)：基础跟踪、单手支持、自主换手、多圈操作与部署学生的阶段和验收目标。
- [当前双手简化任务](manipulation/docs/HAND_ONLY.md)：双手接触物理与独立H1运动学可达性守卫。
- [双手任务设计](manipulation/docs/HAND_CENTRIC_DESIGN.md)：回正负载下持续承载、换握与双向跟踪的网络及课程建议。
- [baseline入口](baseline/README.md)：原驾驶演示和评价。
- [manipulation入口](manipulation/README.md)：臂手操作、训练和评价；含原固定抓握残差实验与新的换手研究入口。
- [第一阶段实际工作记录](manipulation/results/regrasp_stage1/README.md)：物理路线审计、教师PPO训练、冻结对照与尚未通过的指标。
- [两方案差异](manipulation/docs/COMPARISON.md)。
- [目录与迁移说明](manipulation/docs/WORKSPACE_MAP.md)。

原固定抓握方案完成了seed7的握持及±10°开发试验。新`regrasp`方案正在执行152维观测、24维臂手联合动作的教师/历史学生路线；物理脚本已验证可用的新抓握，但策略尚未通过稳定握持/自主换手验收。当前不接上游、CARLA或实物。已放弃的旧速度PPO版本保持删除。

## 从根目录运行

`run.sh`自动进入对应项目、配置其源码路径并使用本机mujoco环境，不需要重新安装。参数中的路径相对于对应项目目录。

```bash
bash manipulation/scripts/run.sh -m wheel_manipulation.evaluate \
  --model runs/implicit_reach10_s7_pilot/best/best_model.zip \
  --mode sine --amplitude 10 --seconds 10 --output outputs/recheck_new

bash baseline/scripts/run.sh -m drive_baseline.evaluate \
  --mode sine --amplitude 10 --seconds 10 --output outputs/recheck_new

bash manipulation/scripts/run.sh -m unittest discover -s tests -v
bash baseline/scripts/run.sh -m unittest discover -s tests -v
```

项目可以分别安装：`pip install -e ./baseline`、`pip install -e ./manipulation`或`pip install -e ./unified_steering`，本机也可使用各自运行脚本。依赖各自在`pyproject.toml`声明。机器人源模型仍通过`H1_MODEL_PATH`或现有Unitree路径提供；新路线场景快照的网格路径见其资产说明。训练权重、原始输出及第三方依赖保留本机，不纳入Git。
