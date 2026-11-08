# 验证

当前场景的几何、装配及初始接触检查入口：

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.audit_scene
```

报告保存在 `results/logitech_scene_v1/audit.json`，失败返回非零退出码。它验证抽样轮圈拟合、镂空、被动轮轴和初始接触，不代表完成转向技能验收。

后续按需增加接口/单位、参考时间推进、策略记忆reset与恢复、数据序列边界和动作映射检查。


仿真P0的动作边界、接近路径、任务连续性和评价器回归：

```bash
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh -m unittest discover -s tests -p 'test_p0*.py' -v
```

历史记录为14项通过；本轮全量发现旧`test_p0_motion`仍依赖当前缺失的历史`continuous_plan`模块，不能宣称当前旧套件全部通过。该历史问题不涉及新层1控制器，原归档保留。包括反向负载力矩不能冒充承载、旧手未离开不能通过单手阶段、停止轮角参考不能冒充运动交接，以及提前终止后未访问的阶段不能使评价器崩溃。物理能力以[P0逐例轨迹](../results/p0_complete_20261107/README.md)为准，单元测试不替代物理验收。

掌心主驱层1的独立物理核对与接力任务不变量：

```bash
OPENBLAS_NUM_THREADS=1 bash scripts/run.sh -m unittest discover -s tests -p 'test_palm_*.py' -v
```

7项检查覆盖世界接触力矩／广义力矩一致性、包含干摩擦和转子惯量的轮轴平衡、独立Jacobian滑移速度，以及累计轮角、混合反向、上半圆窗口、接触等待不改变已执行运动和接管后独立保持。完整物理结论以[新层1报告](../results/physical_feasibility/l1_v1/README.md)为准；一键矩阵见[工程系统说明](../docs/LAYER1_ENGINEERING.md)。
