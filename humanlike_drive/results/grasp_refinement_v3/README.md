# 第3版抓握：让无名指和小指支撑轮圈

2026-11-06。修正1号抓握的无名指、小指压在后方金属拨片上的问题。原机器人、G29可见模型、拨片碰撞以及mimic传动关系均保留。

[侧后方前后对照](before_after_grip_rear.png) · [正面前后对照](before_after_driver.png) · [四指实际轮圈接触标注](annotated_rim_contacts.png) · [检查报告](audit.json)

## 问题与修改

旧统计只要求手指碰到任意 `collision_*` 手轮部件，因此无名指、小指在拨片上的受力被误算为有效抓握。第2版“四指支撑轮圈”的结论已作废，旧图与报告保留在原目录。

本版调整手掌位置、朝向和六个独立驱动的闭合目标，让无名指、小指回到轮圈背面；掌部和拇指也保持轮圈接触。接触报告现在分别列出每根手指的轮圈力和非轮圈受力，查看器以品红色标记后者。

CAD核对还发现，原轮圈单层扇区凸包把下部内侧握持凹面填平，小指处出现约2毫米的虚假接触表面。现在将每个周向扇区按轴向分为5层，总轮圈480块；含拨片、轮辐和轮毂共606块。拨片外侧扇区单独命名并显示为粉色。没有删除、移动或关闭拨片碰撞。

原“手腕到位后再闭合”的准备动作会让无名指扫到拨片。现在在轮圈外侧提前预弯四指：从张开状态开始，前0.6秒完成预弯，手腕在1秒内接近，拇指稍后闭合。总准备6秒，临时轮轴居中辅助在2.5秒撤除。

每只手仍为6驱动。四指第二关节跟随近端1倍；拇指中间、末端跟随屈曲驱动1.6倍、2.4倍，均从配套H1-2 URDF导入。未独立调整从动关节。

## 实测检查

各预设保存后重新载入，额外保持5秒，轮轴无执行器和直接辅助力矩。轮圈接触以单个接触点法向力超过0.05 N为准，按50 Hz共采样250次；下表取参与抓握手的最小值。

| 预设 | 四指各自轮圈接触占比 | 掌部／拇指轮圈占比 | 非轮圈受力 | 最小手—拨片间隙* | 最大接触点—原CAD误差* | 最小双臂关节余量 |
| --- | --- | --- | --- | --- | --- | --- |
| 1：9 / 3 | 均100% | 均100% | 0 N | 4.01 mm | 0.68 mm | 14.27° |
| 2：10 / 2 | 均100% | 均100% | 0 N | 4.30 mm | 0.61 mm | 6.02° |
| 3：左手支撑 | 均100% | 均100% | 0 N | 4.03 mm | 0.68 mm | 7.84° |
| 4：右手支撑 | 均100% | 均100% | 0 N | 4.07 mm | 0.30 mm | 7.84° |

\* 几何间隙及原CAD三角面距离在保持初、末状态测量；不是整个运动过程的全局误差上界。准备阶段另以50 Hz检查非轮圈受力，四个预设均未检测到超过0.05 N的接触。保持阶段每个物理步都检查非轮圈加载。

1号保持阶段，无名指的最小轮圈法向力为左0.77 N、右0.83 N，小指为左0.74 N、右0.79 N。最大轮角漂移0.023°，最大mimic偏差0.018°。所有预设初始穿透为0，准备过程最大穿透0.301 mm以内；29项检查通过。原CAD外轮廓抽样95%误差0.018 mm，最大1.47 mm，未漏检。

这些结果验证初始接触与短时保持，尚未验证外部回正负载、连续转向和实物表现。机器人自碰撞响应仍关闭，静态间隙检查不能替代运动过程的自碰撞处理；摩擦、传动和执行器参数仍需实物标定。

## 查看与重建

```bash
cd /home/nubot/humanlike_drive/humanlike_drive
bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene --camera grip_rear
```

`1`为本次重点修正的9/3抓握；`V`切换相机，新增左手侧面和背面；`T`显示半透明受力点，非轮圈接触为品红色；`C`查看碰撞体，绿色为轮圈、粉色为拨片；空格运行／暂停，`R`重置。

目标配置：`humanlike_drive/config/controller/grasp_targets.json`。修改前配置和状态在本目录的 `before_targets.json`、`before_grasps.json`。

```bash
bash scripts/run.sh -m humanlike_drive.env.logitech_scene
bash scripts/run.sh -m humanlike_drive.controller.basic_grasps
bash scripts/run.sh -m humanlike_drive.env_runner.audit_scene --output results/grasp_refinement_v3/audit.json
MUJOCO_GL=egl bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene --render results/logitech_scene_v1
```
