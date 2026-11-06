# 1号抓握：让前臂与手掌自然衔接

2026-11-06，配置第5版。1号双手9点／3点抓握已从约35°腕部内扣调整为约5.3°，前臂与手掌夹角约9°。2–4号的完整保存预设与第4版逐项一致。

[正面前后对照](before_after_driver.png) · [俯视前后对照](before_after_hands.png) · [驾驶室对照](before_after_cabin.png) · [最终背面](after_grip_rear.png) · [角度数据](pose_changes.json) · [完整检查报告](audit.json)

参考了Škoda官方驾驶员握持照片和Conquer Driving驾驶姿势教学视频的实际抽帧，主要观察前臂与手掌如何衔接、肘部如何弯曲。[参考来源与抽帧](references/README.md)提供原始链接和视频时间点。

上一版保留手掌朝向，仅通过肩肘重新分配姿态；该朝向让腕部仍明显内扣。这次允许1号的手掌朝向、接近位置与六驱动目标一起调整，再求解肩肘姿态。手掌位置稍向外侧调整，改变包络角度，让掌部、拇指与四指共同贴住真实轮圈，同时给后方拨片留出空间。

| 指标 | 修改前 | 第5版 |
| --- | --- | --- |
| 准备完成时腕部yaw绝对值 | 双腕约35.2° | 左5.29°／右5.33° |
| 额外保持期间腕部yaw最大绝对值 | — | 左5.29°／右5.34°以内 |
| 额外保持期间前臂与手掌最大夹角 | — | 左9.20°／右9.37° |
| 四指、掌部、拇指的轮圈接触占比 | — | 双手均100%，250次／50 Hz采样 |
| 非轮圈法向受力 | — | 准备采样及保持检查均为0 N |
| 手到拨片间隙 | — | 保持前后静态检查均≥15.1 mm |
| 实际轮圈受力点到原CAD表面误差 | — | 保持前后检查最大≤0.208 mm |
| 保持期间最小手臂机械关节余量 | — | 4.29° |
| 保持期间轮角最大漂移 | — | 0.0098°以内 |

腕部yaw是原模型腕偏关节角；前臂与手掌夹角按 `wrist_roll_link` 与 `wrist_yaw_link` 的纵向X轴夹角计算，用于补充检查pitch造成的弯折。这些是机器人运动学指标，未从参考照片测量人体角度。保持阶段每个物理步检查腕偏≤10°、前臂与手掌夹角≤15°，防止将内扣简单转移到另一腕轴。

原有29项几何、接触与保持检查全部通过，新增1号腕姿检查后共30项。额外保持5秒无轮轴辅助：每根手指轮圈法向力最小约0.30–0.66 N，四指持续支撑；掌部和拇指也持续接触。准备过程最大穿透0.091 mm，保持阶段无穿透；从动误差最大约0.016°。非相邻手指保持结束时最小静态间隙2.78 mm，双臂及前臂／手与躯干的静态检查最小间隙31.8 mm，无检测到的重叠。

没有修改机器人原始运动学、关节限位、六驱动mimic、场景几何或执行器增益。拨片保留真实外观与碰撞。机器人自身碰撞响应仍沿用当前固定坐姿设置，静态距离检查不能替代运动过程的完整自碰撞验证。本次只完成初始抓握和5秒保持检查，尚未验证连续转向、外部回正负载或实物效果。

窗口查看：

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene --preset nine_three --camera driver
```

按 `1` 看本次姿态，`V` 切换相机，`T` 看实际受力点，空格运行／暂停。配置位于 `humanlike_drive/config/controller/grasp_targets.json` 的 `presets.nine_three`。修改前的配置和抓握保存在本目录 `before_targets.json`、`before_grasps.json`。

生成与验证：

```bash
bash scripts/run.sh -m humanlike_drive.controller.basic_grasps
bash scripts/run.sh -m humanlike_drive.env_runner.audit_scene --output results/driverlike_grasp/audit.json
MUJOCO_GL=egl bash scripts/run.sh outputs/driverlike_grasp/render_compare.py
```

本次联合拟合、接触微调的脚本与中间候选放在 `outputs/driverlike_grasp/`；正式场景和查看器只读取已保存的第5版配置及抓握。
