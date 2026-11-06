# 闭环评估执行器

当前 `inspect_scene.py` 提供交互窗口和静态图集，`audit_scene.py` 检查轮圈几何、装配及四种预设的短时接触。命令和验证边界见[场景说明](../../docs/SCENE_LOGITECH_G29.md)。

参考Diffusion Policy的env_runner职责，准备驱动冻结策略运行完整任务、控制reset与策略记忆、逐例汇总误差/事件/执行边界指标并导出视频。开发、回归与独立评估使用显式协议；原始产物写outputs，正式结论写results。
