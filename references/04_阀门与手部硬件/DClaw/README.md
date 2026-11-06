# Dexterous Manipulation with Deep Reinforcement Learning: Efficient, General, and Low-Cost

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/1810.06045) · [公开来源](https://arxiv.org/abs/1810.06045)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

目标角度条件的低成本真实RL环境，是检查轮角观测、角度误差和上一动作接口的简洁参照。

## 对当前方向盘任务的借鉴

参考目标条件观测与任务成功门槛的明确写法；用简单任务隔离手部接触学习。

## 限制与区别

三指九驱动小阀门与包络大轮盘不同。周期角度误差会折返；多圈或有限行程任务必须另外表达进展与限位。

## 建议怎么读

先看原论文任务与真实学习；再读ROBEL turn.py的默认观测、角度包裹、奖励和成功阈值，避免混用两套定义。

## 此前阅读记录中的方法与实验细节

##### DClaw / ROBEL

- [2018 论文](https://arxiv.org/abs/1810.06045)
- [ROBEL](https://github.com/google-research/robel)
- [目标角度任务代码，固定版本](https://github.com/google-research/robel/blob/5b0fd3704629931712c6e0f7268ace1c2154dc83/robel/dclaw/turn.py)

三指九驱动的小型阀门操作，原论文研究模型自由强化学习和示教增强，包含真实学习。ROBEL 提供目标角度误差、上一动作、手指位置等观测；观测字典另有速度等信息，但不能认定全部字典项都被默认拼入策略。读取的版本默认选择 `claw_qpos, object_x, object_y, last_action, target_error`。

动作控制手指；angle error 被包裹到 [-π, π]。这适用于周期角度到达任务。若方向盘任务需要记录多圈进展，应另外保留不折返的角度/圈数，不能仅用 sin/cos 与最短角度差区分目标。ROBEL 默认成功阈值 0.1 rad 约 5.7°，不是高精度跟踪保证。ROBEL 与原论文的奖励阈值也不应混写。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/valve_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
