# 工作区导航与项目隔离

根目录下保留`baseline/`和`manipulation/`两个任务项目，论文统一放在`references/`。根README是索引，Git历史仍位于根目录。

- `baseline/src/drive_baseline/`：原驾驶源码；数据、权重、CARLA和安装包也在baseline内。
- `manipulation/src/wheel_manipulation/`：当前源码；接触/渲染/硬件基础模块在其内部common中。
- `references/`：按六个主题分类的论文库，每篇独立文件夹包含原文或链接、理解与阅读建议；检索缓存和历史资料在隐藏的`.archive/`内，不参与训练执行。
- 两项目各有自己的`pyproject.toml`、`assets/`、`configs/`、`scripts/`、`tests/`、`runs/`、`outputs/`、`results/`与`docs/`。

在根目录用`bash baseline/scripts/run.sh ...`或`bash manipulation/scripts/run.sh ...`。脚本进入对应任务目录，因此参数里的`runs/`和`outputs/`都只指该任务。也可以进入任务目录安装其editable包。

原`humanlike_drive/baseline/`移到`baseline/src/drive_baseline/`，原`humanlike_drive/manipulation/`移到`manipulation/src/wheel_manipulation/`；外层统一源码包已移除。原`third_party/`移入baseline。论文资料曾随项目隔离移入manipulation，2026-10-23按用户要求统一移回根目录`references/`。

[隔离迁移清单](../results/isolation.json)记录最新对应关系；[此前整理清单](../results/reorganization.json)记录速度PPO删除和旧v2产物迁移。原始实验JSON、checkpoint、场景均保留字节和历史来源，不将旧路径改写成新的实验来源。

当前模型在manipulation的`runs/implicit_reach10_s7_pilot/best/`，驾驶视频在baseline的`outputs/carla_drive_clockwise/drive.mp4`。当前仍只完成±10°开发试验。

[当前任务入口](../README.md)、[差异说明](COMPARISON.md)、[pilot结果](../results/pilot/README.md)、[论文总索引](../../references/README.md)、[sim2real论文](../../references/03_sim2real/README.md)。
