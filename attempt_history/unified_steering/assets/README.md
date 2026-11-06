# 场景资产

`scene.xml` 由 `unified_steering.audit prepare` 从显式指定的现有 XML 冻结，`scene.provenance.json` 保存来源及哈希。两者是本机生成物，不提交 Git；报告保留完整来源信息。

默认选用 `../../manipulation/assets/regrasp_stage1/wheel.xml`，其中跨指碰撞比旧固定抓握场景更完整。这里只复用模型，不导入旧任务环境、控制器、奖励或策略。网格仍是外部 Unitree 资源，首次准备可用 `H1_MESH_DIR` 指定位置；运行前复核哈希。

场景仍含原模型的简化臂碰撞守卫和既有接触假设。有限力矩不代表已标定，碰撞几何不构成完整安全覆盖。轮轴原有限位保留，不能把累计角度表示误当成无限行程。

## 精选初态

[S00-B初态目录清单](initial_states/s00_support_v1/manifest.json)包含10个经过保持检查的单手支持状态，左右各5个，轮角在+30°附近，自由臂关节扰动±2°。状态与来源哈希、验证轨迹绑定。该小型精选库随代码保存；原始输出仍放在outputs。恢复时必须清除外部辅助力矩，不把reset准备夹具带入任务步。范围与结果见[S00-B报告](../results/S00_SUPPORT.md)。
