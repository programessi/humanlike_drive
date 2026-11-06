# 方向盘操作与机器人驾驶：初步文献核对

后续专项调研：[连续旋转、接触切换与双臂双手换握（2026-10-22）](contact_switching_review/REVIEW.md)、[双手旋拧 RL](bimanual_rl_review/REVIEW.md)、[manipulation sim2real](manipulation_sim2real_review/REVIEW.md)、[阀门与 Inspire 手](valve_review/REVIEW.md)。下文保留 2026-10-03 的初步核对与当时实现描述。

检索日期：2026-10-03。使用 OpenAlex 公开元数据检索，并下载、阅读 Musashi 驾驶论文相关章节。此记录不是系统综述，检索未发现完全相同的组合不能作为“首次”的证据。

| 工作 | 已核实内容 | 与项目的关系 |
| --- | --- | --- |
| Toward Autonomous Driving by Musculoskeletal Humanoids: A Study of Developed Hardware and Learning-Based Software，2020，IEEE Robotics & Automation Magazine。DOI: https://doi.org/10.1109/MRA.2020.2987805；公开稿 https://arxiv.org/abs/2406.05573 | Musashi 坐在真实车辆内操作踏板和方向盘。软件包括在线学习的关节角/肌肉张力/肌肉长度关系、学习的任务动力学、反射控制及视觉/声音识别。论文说明方向盘实验采用手部压接与 push-pull 操作，抓握和长时间耐久测试仍有不足。其研究不是本项目拟议的 model-free PPO 双臂/手指操作策略。 | “人形机器人通过手操作方向盘驾驶”已有前作。应比较持续抓握方式、动作接口、学习方法和驾驶评估。公开稿虽在 2024 年上传，正式论文年份为 2020。 |
| An Admittance Control Method Based on Parameters Fuzzification for Humanoid Steering Wheel Manipulation，2023。https://doi.org/10.3390/biomimetics8060495 | 公开摘要说明：7-DOF 仿人机械臂、模糊参数导纳控制、CoppeliaSim 仿真，研究方向盘操作中的柔顺受力与极端构型。本文仅核对元数据和摘要，未阅读全文。 | 可作为几何/力控制方向盘操作的相关工作；不应把普通方向盘操作或柔顺控制声称为首次提出。 |
| Dexterous Manipulation with Deep Reinforcement Learning: Efficient, General, and Low-Cost，2018 预印本。https://arxiv.org/abs/1810.06045 | 摘要说明多指手通过 model-free 深度 RL 直接从传感器状态学习动作，在真实硬件上学习多种接触丰富的操作任务。本文仅核对元数据和摘要。 | 纯 RL 学习接触操作已有成熟研究；本项目的潜在研究贡献需要针对持续方向盘操作与道路驾驶闭环进一步定义。 |

## 当前实现与拟议范式的区别

当前实现：图像 → 视觉网络预测道路 XY 路径点 → 几何路径控制器生成轮盘目标角 → 角度反馈/圆弧参考 → IK + PPO 关节残差 → PD → 物理接触 → 实测轮盘角 → CARLA。

现有 PPO 由强化学习训练，但执行依赖圆弧 IK 基础参考，准确名称是 residual RL。其网络输入没有道路路径点；路径点先转换成目标轮盘角。因此不能把现有 checkpoint 描述为无 IK 的纯 RL 操作策略或直接 waypoint-conditioned 策略。

若采用目标角条件的纯 RL 操作层，可保留道路路径点到目标角的几何控制器，替换“圆弧参考/IK + PPO 残差”为“机器人状态 + 轮盘目标角/角速度 → RL → 双臂与手指关节目标”。保留关节 PD 和执行器限幅并不妨碍策略属于 RL。

若采用直接路径点条件策略，则输入“道路路径点 + 车速 + 机器人/轮盘状态”，输出双臂与手指关节目标，去除手部圆弧 IK 和外部路径点到轮盘目标角的执行链。该策略需新的训练任务和驾驶奖励；目前尚未实现或训练。

潜在贡献应通过实验建立，例如持续双手抓握、滑移恢复与重新抓握、跨轮盘安装几何泛化、理想车辆转向与物理机器人转向的性能差异。它们均不是当前已经完成的研究结果。
