# 研究文档

当前按三层组织研究，场景保留，已有尝试凝练为经验；层1工程系统已实现，验收结论按实际运行报告。

- [层1验收报告与视频](../results/physical_feasibility/l1_v1/README.md)、[非学习系统手册](LAYER1_ENGINEERING.md)、[工程选择记录](decisions/2026-11-07_L1_ENGINEERING.md)。
- [总计划v4.2](plans/RESEARCH_PLAN.md)：物理可行性 → 现有方法适用性 → 针对性提升，定义任务、证据、阶段出口与论文定位；仿真层1代表±180°往返已完成。
- [历史经验](experiments/HISTORICAL_LESSONS.md)：拨片误接触、碰撞近似、腕姿、动态承载、重复接管与评价边界，均链接原记录。
- [第一层物理协议v1.3](experiments/PALM_PRIMARY_P0.md)：9点／3点自然腕姿、掌心摩擦主驱与手指顺应、两手上半圆干涉检查，优先0—180°往返；360°／540°可选扩展，成功轨迹另做专家数据检查。
- [第二层评价协议](experiments/EXISTING_METHODS_EVALUATION_V1.md)：适配卡、方法筛选、数据／观测／预算、公平比较与失败诊断。
- [第二层候选B0草案](plans/BASELINE_IMPLEMENTATION_V1.md)：BC初始化、PPO特权教师与历史学生；供筛选，已不作为默认执行路线。
- [R0实物入口](plans/R0_HARDWARE_CALIBRATION.md)：接口、几何、负载、反馈、时序和行程标定；实物对应三层逐级验证。

可复用资产与历史结果：

- [G29坐姿驾驶室](SCENE_LOGITECH_G29.md)：现有场景、查看器与模型来源；本次没有重建或改动。
- [90°独立备份](../results/archive/20261107_pre_palm_primary/README.md)：原始轨迹、完整视频、配置、源码和校验清单，仅作历史参考。
- [旧动态P0](../results/p0_complete_20261107/README.md)：旧路线往返、承载、交接、连续序列和失败边界，不作为新路线完整可行性证据。
- [抓握第3版](../results/grasp_refinement_v3/README.md)、[腕部调整](../results/wrist_refinement/README.md)、[自然腕姿](../results/driverlike_grasp/README.md)：保留原条件和限制，经验统一在上文汇总。

旧方案供追溯：[v4.0](plans/RESEARCH_PLAN_v4_0_before_upper_semicircle.md)、[v3.1及当时B0优先级](plans/RESEARCH_PLAN_v3_1_before_three_layers.md)、[v3.0](plans/RESEARCH_PLAN_v3_0_before_relay.md)、[v2.3](plans/RESEARCH_PLAN_v2_3_before_palm_primary.md)、[历史初版](plans/RESEARCH_PLAN_v1_before_ten_two.md)。历史决策按原日期保留，执行优先级以当前总计划为准。

论文库继续使用父工作区references。每项实际试验开跑前保存案例、门槛、预算、停止条件和数据／代码版本，不以计划中的网络或成功率目标冒充已有结果。
