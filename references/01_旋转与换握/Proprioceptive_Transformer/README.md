# Learning Robust Dexterous In-Hand Manipulation from Joint Sensors with Proprioceptive Transformer

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2605.21330) · [公开来源](https://arxiv.org/abs/2605.21330)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

关节历史可用于推断瞬时不可见的交互状态；教师蒸馏加状态重建，比单纯把MLP换成Transformer有更明确的学习目标。

## 对当前方向盘任务的借鉴

待教师具备合格操作能力后，比较历史学生的动作拟合、隐变量估计及状态重建；保留命令和实际执行差。

## 限制与区别

ORCA手加装直接关节角传感器。Inspire主动驱动反馈不等于每个联动指节的真实位置，观测可用性必须重新核对。

## 建议怎么读

先读真实传感配置；再画出teacher/student输入差异；最后看重建损失、历史窗口和传感/策略消融。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/contact_switching_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
