# In-Hand Manipulation of Articulated Tools with Dexterous Robot Hands with Sim-to-Real Transfer

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2509.23075) · [公开来源](https://arxiv.org/abs/2509.23075)

阅读优先级：优先阅读。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

六驱动耦合手可以操作关节工具，但可行初抓握、机械接触改造与真实适配都是最终能力的一部分。

## 对当前方向盘任务的借鉴

参考六驱动加mimic的动作/观测边界、示教可行初态和仿真基础技能；先验证本机手的真实接口。

## 限制与区别

实物含触觉阵列、驱动负载、手套及部分工具适配，CATFA用真实数据。不能描述成裸Inspire只靠仿真PPO完成迁移。

## 建议怎么读

先看oracle与本体学生输入；再看仿真执行器和初抓握；重点核对CATFA训练数据及手部/工具硬件改造。

## 此前阅读记录中的方法与实验细节

#### 6. 更贴近 Inspire 实物的补充：关节工具操作

**In-Hand Manipulation of Articulated Tools with Dexterous Robot Hands with Sim-to-Real Transfer**，Soofiyan Atar、Daniel Huang、Florian Richter、Michael Yip。初稿 2025-09；读取 v3，2026-03-05 更新。

- [论文](https://arxiv.org/abs/2509.23075)

它使用 **六驱动 Inspire + 六联动指节**，从可行握持构型开始，学习剪刀、钳等内部关节开闭，同时避免物体姿态漂移。目标包含关节状态，并非仅奖励一直往一个方向转。PPO oracle 输出六维绝对手关节目标，EMA 系数 0.5。

仿真 oracle 读取六驱动 q/dq、上一目标、工具关节角/角速度、工具轴参考坐标、命令和模拟关节力；随后蒸馏为只读取本体位置与开闭命令的 student。最后使用真实示教、触觉与驱动负载训练 CATFA 适配器，改善未模拟的摩擦、卡滞与受扰响应。

实物增加了 36×44 触觉阵列、接触垫/泡沫、丁腈手套，并有主动关节力读数；部分工具增加机械适配件。真机 30 Hz 目标更新；仿真回合约 50 s。可行初态在必要时来自 Manus 手套示教。

它证明低驱动数手也值得研究关节物体操作，但不能作为“裸 Inspire 双手仅用仿真无示教 RL 已经解决方向盘”的证据。它提供另一条迁移路线：**先学仿真基础操作，再用真实传感闭环补足接触差异**。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/bimanual_rl_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
