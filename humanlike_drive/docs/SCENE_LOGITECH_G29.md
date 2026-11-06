# G29 坐姿驾驶室与基础抓握

更新日期：2026-11-06。这一版先解决场景、几何和初始接触，让人可以直观看布局。尚未训练转向策略，也没有连接实物。

## 直接查看

```bash
cd /home/nubot/humanlike_drive/humanlike_drive
bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene
```

窗口初始暂停，默认使用驾驶室相机。

| 按键 | 作用 |
| --- | --- |
| `1` | 双手9点／3点 |
| `2` | 双手10点／2点 |
| `3` | 左手接触，右手退到旁边 |
| `4` | 右手接触，左手退到旁边 |
| `V` | 循环切换驾驶室、整体、驾驶员视野、手部、抓握正面、方向盘、左手侧面、左手背面 |
| `F` | 自由相机，可用鼠标拖动、缩放 |
| `C` | 切换外观／碰撞体视图：轮圈绿色，后方拨片粉色，轮辐轮毂橙色，机器人蓝色 |
| `T` | 半透明显示轮圈受力点：掌部蓝色、拇指黄色、近端指节绿色、末端红色；非轮圈接触统一为品红色 |
| 空格 | 运行／暂停物理仿真 |
| `R` | 重置当前抓握并暂停 |

指定视角或抓握，例如：

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene --camera driver_eye --preset ten_two
```

无需MuJoCo图形窗口也可[查看图集](../results/logitech_scene_v1/index.html)。其中包括[驾驶室全景](../results/logitech_scene_v1/nine_three_cabin.png)、[驾驶员视野](../results/logitech_scene_v1/nine_three_driver_eye.png)、[四种抓握](../results/logitech_scene_v1/grasps_overview.png)和[碰撞体对照图](../results/logitech_scene_v1/g29_collision_overlay.png)。碰撞视图隐藏重合的原始CAD表面，避免两层表面造成闪烁。

## 场景里有什么

- 机器人采用原始 H1-2 与 Inspire 双手模型，保留手臂、手指的运动学与关节限位。髋、膝分别弯曲约90°，下半身、骨盆和躯干固定，形成坐姿；当前不研究全身平衡。
- G29 外壳水平放在安装台上。没有额外旋转底座来调整手轮倾角，约24.07°的转轴倾角来自原URDF。
- 安装台面高0.94 m，轮轴原点约为 `(0.548, 0, 1.037) m`。模型主要底部支撑面贴住台面，前方夹具从台沿垂下。机器人骨盆高度0.83 m。
- 驾驶室包括座椅、头枕、侧翼、座椅导轨、仪表台、出风口、中央扶手与挡杆、杯架、脚踏板、油门和刹车，以及两侧舱壁、A柱、后柱和后视镜。顶部留空，方便检查机器人与方向盘。
- 前方大屏的显示区域约1.44 × 0.71 m。道路和仪表是程序生成的静态图片，目前没有接入CARLA，也不表示车辆正在运动。

这是一套便于研究检查的驾驶模拟器座舱；座椅和舱体是程序化简化模型，不对应某款实车的扫描尺寸。

## 模型来源与碰撞处理

本轮重新读取以下源文件，没有导入旧场景或旧抓握快照：

| 资源 | 本机来源 |
| --- | --- |
| H1-2和双手 | `/home/nubot/unitree/unitree_ros/robots/h1_2_description/h1_2.xml` 及其 `meshes/` |
| G29 | `/home/nubot/g29_wheel_urdf/urdf/g29_wheel_urdf.urdf` |
| G29外壳／手轮网格 | `base_link_wheel.STL`／`Link1.STL` |

原G29 URDF引用的外壳文件缺失，碰撞字段还存在拼写错误。本版使用同目录实际存在的外壳CAD，保留原轮轴装配变换。来源哈希和装配信息存放在[provenance.json](../assets/objects/logitech_g29/provenance.json)。

MuJoCo会将单个碰撞mesh按凸体处理，直接使用完整手轮会把镂空填住。轮圈沿周向拆成96个扇区，第3版再按轴向分为5层，保留下部内侧握持凹面，避免原单层凸包将小指附近表面外扩约2毫米。加上按径向、周向和深度分块的拨片、轮辐与轮毂，共606个凸块。分块使用原CAD顶点；后方拨片有单独名称和颜色，保留实际碰撞。外壳目前仍用保守凸包，未做精细分解。

轮轴为被动关节，没有轮轴电机。当前限位为±450°，即G29常见的900°总行程；从居中零位单向转540°会碰到止挡，后续研究该目标前必须先确定实物与角度定义，不能直接扩大限位冒充G29能力。

## 抓握如何生成

第3版修正了拨片误接触：第2版无名指和小指实际上压在后方拨片上，旧统计误把它们计入轮圈支撑。现在分别验证每根手指在轮圈上的受力，并排除拨片、轮辐与轮毂接触。没有修改原mimic比例。接近时四指提前预弯，避免到位后再从张开状态闭合而扫过拨片；拇指随后轻触轮圈。

当前配置为第5版。第4版固定手掌朝向，仅调整肩肘，1号腕偏仍约35°。本次参考真实驾驶员照片与教学视频抽帧，为1号重新联合设计手掌位置／朝向、六驱动闭合量和肩肘姿态，物理保持时双腕内扣约5.3°、前臂与手掌夹角约9.2–9.4°。四指、拇指和掌部在5秒保持中持续支撑轮圈，拨片无受力；2–4号预设与第4版完全一致。`arm_posture` 中的腕偏和肘部位置是IK偏好，不改变机械关节限位；1号另设实际腕偏≤10°、前臂与手掌夹角≤15°的检查门槛。这些角度是本模型的工程目标，未从人类照片中反推关节角。见[本次调整、参考来源与对照](../results/driverlike_grasp/README.md)。

准备共6秒，其中1秒从外侧接近，四指在接近时预弯；2.5秒时移除临时轮轴居中力矩，继续稳定3.5秒。初始穿透、准备穿透和非轮圈受力均纳入检查。导出的场景和查看器没有轮轴辅助，只使用手臂偏置力补偿和有限电机命令。

保存物理闭合后的实际平衡状态，没有强行把轮角改回零；各预设轮角见 `assets/scenes/basic_grasps.json`。这些仍是基础接触预设，尚未验证回正负载下的承载、换握或指令跟踪。

### mimic从动关系

本版从与当前手部几何配套的 H1-2 URDF 自动导入从动定义，复制为 `assets/robots/h1_2/source.urdf`。同一份定义同时用于MuJoCo等式约束、初始关节赋值和驱动限位推导。

| 从动关节 | 关系 |
| --- | --- |
| 食指／中指／无名指／小指的第二关节 | `q_second = q_proximal` |
| 拇指中间关节 | `q_intermediate = 1.6 × q_pitch` |
| 拇指末端关节 | `q_distal = 2.4 × q_pitch` |

每只手保持6个独立驱动，没有把从动关节改成独立控制。拇指屈曲驱动的联合有效范围为 **0–0.5 rad**：虽然主动关节单独限位为−0.1到0.6 rad，但两个从动关节的0–0.8、0–1.2 rad限位会进一步约束它。查看器和预设生成都遵守这个联合范围。

这些比例与当前H1-2 URDF一致，不等于已标定真实机械传动。本机其他FTP／DFQ模型有不同几何和从动参数，不能只借用它们的比例套到本模型上。实物阶段仍需根据实际手型号测量。

为减少手指闭合反力把手腕推出轮圈的现象，手臂位置增益提高为原来的2倍，阻尼相应提高；电机输出上限不变。手臂、手指的参数仍是仿真工程初值。受力接触统计以法向力超过0.05 N为准；近端指节只在部分姿态参与，未强求每个指节同时贴住轮圈。

拨片接触修正见[第3版抓握报告](../results/grasp_refinement_v3/README.md)，当前腕部姿态和5秒保持检查见[第5版调整记录](../results/driverlike_grasp/README.md)。第2版图像和报告保留作历史记录，其中“四指支撑轮圈”的结论已更正。

手指耦合、摩擦、轮轴阻尼、执行器增益和输出上限目前为工程初值，后续用实物测量更新。当前固定坐姿查看场景关闭了机器人自身的碰撞响应；机器人与方向盘、座舱的外部碰撞开启。另做了双臂之间、前臂和手与躯干之间的静态距离检查；这不能替代后续运动过程的完整自碰撞验证。

## 验证与重建

运行几何和接触检查：

```bash
bash scripts/run.sh -m humanlike_drive.env_runner.audit_scene
```

检查内容包括底座水平、自然转轴倾角、轮轴无执行器、轮圈1800条射线拟合、两个上方镂空点，以及四种预设额外5秒无轮轴辅助保持。按50 Hz分别统计四指的轮圈接触、掌部／拇指轮圈接触与非轮圈受力；额外比较实际受力点到原CAD三角面的距离，并检查手到拨片的几何间隙。还检查从动误差、关节余量、穿透、轮角漂移、异常碰撞和非相邻手指静态间隙；每个物理步检查1号的腕偏和前臂与手掌夹角，防止只在截图时姿态自然。当前30项检查通过，失败返回非零退出码。见[第5版 audit.json](../results/driverlike_grasp/audit.json)。

当前轮圈抽样95%误差约0.018 mm，最大约1.47 mm，没有轮圈漏检，抽样镂空点保持畅通。这个结果仅针对指定深度与方向的抽样外轮廓，不是完整三维误差上界。详细接触结果以报告为准；这些检查不包含外部负载能力、转向轨迹或真实设备实验。

修改几何或安装参数后，按顺序重新构建和生成抓握：

```bash
bash scripts/run.sh -m humanlike_drive.env.logitech_scene
bash scripts/run.sh -m humanlike_drive.controller.basic_grasps
bash scripts/run.sh -m humanlike_drive.env_runner.audit_scene
MUJOCO_GL=egl bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene --render results/logitech_scene_v1
```

构建命令支持 `--robot-source`、`--g29-source`、`--table-height`、`--base-x`。单独重建XML会清除keyframe，因此需要随后重新生成抓握。查看现有场景不需要访问外部原始目录。

| 代码 | 职责 |
| --- | --- |
| `humanlike_drive/env/logitech_scene.py` | 导入原始机器人与G29、生成碰撞块、装配场景 |
| `humanlike_drive/env/cabin.py` | 程序化座舱与屏幕预览 |
| `humanlike_drive/controller/basic_grasps.py` | 求解IK、物理闭合、保存四种抓握 |
| `humanlike_drive/env/inspire_hand.py` | 读取原URDF的mimic、推导联合限位、检查从动偏差 |
| `humanlike_drive/config/controller/grasp_targets.json` | 掌部位置姿态、六驱动闭合目标与接近时间 |
| `humanlike_drive/controller/grasp_contacts.py` | 区分掌部、拇指、近端／末端指节的受力接触 |
| `humanlike_drive/env_runner/inspect_scene.py` | 交互查看与离线渲染 |
| `humanlike_drive/env_runner/audit_scene.py` | 几何、接触和短时物理检查 |
