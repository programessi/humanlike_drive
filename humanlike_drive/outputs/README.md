# 单次运行产物

后续每次运行使用唯一目录：`outputs/YYYY.MM.DD/HH.MM.SS_<name>/`。

准备随运行保存完整解析配置、代码版本/快照、数据清单、随机种子、训练或评估日志、metrics、checkpoints、逐例轨迹及视频。若使用Hydra，可沿用其配置快照机制。评估、采集和数据处理运行同样保留来源信息。

正式模型留在对应运行的checkpoints中，由results记录路径与哈希，避免维护多个含义不明的best/final副本。README以外的运行产物默认不入Git。

当前按[三层计划](../docs/plans/RESEARCH_PLAN.md)建议分别使用`physical_feasibility/<run_id>/`、`existing_methods/<method>/<run_id>/`和`targeted_improvements/<method>/<run_id>/`，在实际运行时建立；层1实际运行已保存在`physical_feasibility/`中。`development/`保留未通过与旧碰撞近似探索；`l1_v2_20261107/`、`l1_v3_20261107/`保留分批验收，正式逐例索引见[层1报告](../results/physical_feasibility/l1_v1/README.md)。未通过案例不能作为成功专家数据。已有输出路径保留，不移动或改名。

旧90°完整副本保留在`archive/20261107_pre_palm_primary/`，对应[备份说明和校验清单](../results/archive/20261107_pre_palm_primary/README.md)。原`p0_complete_20261107/`继续保留供历史回放，不作为新主线通过证据。
