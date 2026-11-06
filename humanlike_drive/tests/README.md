# 验证

当前场景的几何、装配及初始接触检查入口：

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.audit_scene
```

报告保存在 `results/logitech_scene_v1/audit.json`，失败返回非零退出码。它验证抽样轮圈拟合、镂空、被动轮轴和初始接触，不代表完成转向技能验收。

后续按需增加接口/单位、参考时间推进、策略记忆reset与恢复、数据序列边界和动作映射检查。
