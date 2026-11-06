# DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2305.05706) · [公开来源](https://arxiv.org/abs/2305.05706)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

关节物体操作可以由PPO输出末端运动和手指目标，再通过Jacobian逆解执行；有IK不代表接触策略没有学习。

## 对当前方向盘任务的借鉴

直接对照当前24维腕手接口；借鉴笛卡尔执行误差、碰撞和阶段奖励诊断，检查目标命令是否真正被执行。

## 限制与区别

22维接口来自xArm6加16驱动Allegro；一次开阀/仿真接触奖励不等于长期包络稳定或真实部署。

## 建议怎么读

先看faucet任务定义；对照base.py动作控制和faucet_env奖励；最后看终止条件、泛化与原论文真实证据边界。

## 此前阅读记录中的方法与实验细节

##### 3.3 DexArt：可读、可复用的抓握转动 RL 任务

**DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects**，2023。

- [全文](https://arxiv.org/abs/2305.05706)
- [代码](https://github.com/Kami-code/dexart-release)
- [水龙头 RL 环境，固定版本](https://github.com/Kami-code/dexart-release/blob/d6ab75e1a0b81384d1ac918cbfd4d97b1f230149/dexart/env/rl_env/faucet_env.py)
- [动作控制实现，固定版本](https://github.com/Kami-code/dexart-release/blob/d6ab75e1a0b81384d1ac918cbfd4d97b1f230149/dexart/env/rl_env/base.py)

任务：抓牢水龙头把手，然后转动约 90°。物体具有实际 revolute joint，不是自由空间画圆。机器人是 xArm6 + Allegro 16 自由度。

论文使用 PPO，PointNet 编码物体/机器人点云，并融合本体状态。22 维动作包括 **掌部六维线速度/角速度 + 16 个手指关节位置目标**。代码通过 Jacobian 逆解将末端速度变成臂关节速度，积分得到位置参考，再使用 PD。因此它是学习策略结合基础运动学控制，不能称为“没有 IK 的纯关节 RL”。

环境代码按接近、抓握、转动给奖励：离把手较远时鼓励接近；要求掌部与至少两根手指接触；抓握建立后奖励转角，并惩罚臂部碰撞、关节速度与笛卡尔控制误差。代码的 `horizon=250`，超过约 1.5 rad 且保持抓握可以提前成功结束。

这份实现适合借鉴阶段奖励、物体关节建模和动作接口。但接触奖励和完成一次开阀，不能直接替代长时间稳定包络抓握指标。接触判断来自仿真，不能默认真实硬件能够获得同样的二值信号。原论文不提供其水龙头策略真实部署的证据。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/valve_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
