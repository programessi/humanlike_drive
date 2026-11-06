# CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2402.14795) · [公开来源](https://arxiv.org/abs/2402.14795)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

真实持续阀门旋转可以由仿真示范增强和动作序列模仿实现，因此接触切换并不排斥模仿学习。

## 对当前方向盘任务的借鉴

若纯RL换握探索长期失败，可将少量可行轨迹通过几何/物理增强扩充；把恢复状态也纳入数据覆盖。

## 限制与区别

固定小型多爪阀门720°完成与大轮辋、无触觉六驱动双手、带时限角度跟踪不同；还有真实示范微调。

## 建议怎么读

先看Rotate任务和原始遥操作动作；再看数据增强、动作分块与时间聚合；最后对照形状泛化和真实微调条件。

## 此前阅读记录中的方法与实验细节

##### 3.4 CyberDemo：真实多圈旋转与示教数据增强

**CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation**，2024。

- [全文](https://arxiv.org/abs/2402.14795)
- [项目与视频](https://cyber-demo.github.io/)
- [项目链接的代码仓库](https://github.com/wang59695487/hand_teleop_real_sim_mix_adr)

Rotate 任务是固定底座、多爪阀门经 revolute joint 旋转 **720°**；用三爪阀门收集数据，再测试四爪、五爪形状。真实机器人为 xArm6 + Allegro。策略读取 RGB 与本体状态，学习遥操作动作序列；使用动作分块、时间聚合和仿真数据增强。全文采集接口描述为六维末端笛卡尔速度与手指关节位置，在几何增强推导中以离散位姿增量描述动作。

有用之处是实际多圈接触操作、形状/位置/视觉条件增强及 sim2real。它没有验证六驱动手对大圆环的包络抓握，也没有要求按外部时间序列逐点达到角度。不能把 720° 完成任务直接等同于驾驶中的低误差动态跟踪。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/valve_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
