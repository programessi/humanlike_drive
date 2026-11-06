# 腕部调整：保留抓握，减轻内扣

2026-11-06，配置第4版。按反馈保留第3版的手掌目标、手指闭合量和mimic，利用双臂IK的冗余调整肩肘姿态，使前臂与手掌衔接更顺。

[正面对照](before_after_driver.png) · [驾驶室对照](before_after_cabin.png) · [姿态变化数据](pose_changes.json) · [检查报告](audit.json)

| 预设 | 主动腕内扣绝对值，调整前 → 后 |
| --- | --- |
| 1：9 / 3 | 约50.45° → 35.19° |
| 2：10 / 2 | 约28.92° → 24.98° |
| 3、4：单手支撑 | 约50.45° → 35.19° |

以上为物理准备后的实际腕部yaw关节角。1号前臂转动约18.5°分担姿态，腕部pitch约3°。未修改轮圈、拨片、机器人关节限位或执行器增益。已逐项核对新旧 `grip_pose` 和 `hand_drives` 完全一致；1号实际平衡后的手掌位置变化小于0.37 mm，朝向变化小于0.67°。

四种预设额外保持5秒，按50 Hz共采样250次；每只主动手的四指、掌部和拇指均保持轮圈接触，未检测到拨片或轮辐受力。准备过程同样未检测到非轮圈受力，最大穿透0.35 mm以内；原有29项检查全部通过。1号轮角最大漂移0.023°，静态手臂与躯干等检查中的最小间隙29.3 mm。

保持原抓握位置时，进一步减小腕偏会使肩关节更接近限位，本次选择约35°。1号保持阶段最小机械关节余量约3°。这仍是初始抓握预设，后续连续转向需要沿轨迹重新分配双臂姿态。

`humanlike_drive/config/controller/grasp_targets.json` 中的 `arm_posture` 指定腕偏目标与左肘位置偏好，右手镜像处理；它们是姿态优化偏好，不改机械限位。退开的手保持原求解方式。

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene --camera driver
```

按 `1` 查看本次重点调整的抓握，`V` 切换相机，`T` 查看接触点。修改前配置与抓握保存在本目录 `before_targets.json`、`before_grasps.json`。

```bash
bash scripts/run.sh -m humanlike_drive.controller.basic_grasps
bash scripts/run.sh -m humanlike_drive.env_runner.audit_scene --output results/wrist_refinement/audit.json
```
