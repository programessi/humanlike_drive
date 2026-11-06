# 研究文档

- [G29 坐姿驾驶室](SCENE_LOGITECH_G29.md)：当前可查看的场景、基础抓握、重建与检查命令。
- [1号自然腕姿调整](../results/driverlike_grasp/README.md)：参考驾驶员照片和教学视频抽帧，联合调整手掌与肩肘，使腕部内扣约5°；当前预设的检查报告。
- [上一轮腕部调整](../results/wrist_refinement/README.md)：第4版仅调整肩肘，1号内扣仍约35°，作为历史记录保留。
- [抓握第3版报告](../results/grasp_refinement_v3/README.md)：修正拨片误接触，分别验证四指轮圈受力、CAD贴合与拨片间隙；第2版的“四指支撑轮圈”结论已作废。
- plans：当前入口为[研究计划第2.1版](plans/RESEARCH_PLAN.md)，以2号10点／2点出发，用统一策略模仿±180°内双向连续追踪和360°／540°换握操作，包含训练与实物实验。[历史初版](plans/RESEARCH_PLAN_v1_before_ten_two.md)保留供追溯。
- decisions：[采用2号的决策](decisions/2026-11-06_TEN_TWO_BASELINE.md)，以及后续确认的任务、接口和方法选择。
- experiments：[下一步2号运动可行性协议](experiments/TEN_TWO_FEASIBILITY_V1.md)，后续每项试验在运行前确定案例、预算、门槛与停止条件。

计划中的网络规模、采样数量和门槛是实施起点；具体实验开跑前在experiments中确定配置与预算。论文库继续使用父工作区references，不复制整套论文。
