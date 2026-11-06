# PatternDex: Learning Interaction Patterns to Guide Reinforcement Learning of Bimanual Dexterous Manipulation of Articulated Objects

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2610.04765) · [公开来源](https://arxiv.org/abs/2610.04765)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

示范可以提供物体运动与腕部/接触关系，而不要求机器人逐关节复刻人手。它给“拟人参考”提供了更合适的层次。

## 对当前方向盘任务的借鉴

未来有示范时，优先提取物体运动、支撑关系和换握顺序；将其作为参考或先验，而非硬绑人体关节角。

## 限制与区别

当前任务没有这类示范数据。参考条件PPO的成功不能证明不依赖示范的自主换握已经解决。

## 建议怎么读

先看interaction pattern怎样从示范提取；再看机器人参考生成与PPO条件；最后核对真实任务与传感依赖。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/contact_switching_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
