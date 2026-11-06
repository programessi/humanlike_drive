# Learning Dexterous Manipulation for a Soft Robotic Hand from Human Demonstrations

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/1603.06348) · [公开来源](https://arxiv.org/abs/1603.06348)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

低驱动数不必然阻止接触操作，机械被动柔顺可以承担部分接触适应；学习与硬件能力要一起分析。

## 对当前方向盘任务的借鉴

用它理解柔顺性如何降低接触敏感度，辅助判断当前电驱手哪些误差可以靠控制适应、哪些源自机械限制。

## 限制与区别

七气动通道软手、物体中心示教和guided policy search；阀门是杠杆且回合内臂固定，不是大轮辋包络旋转。

## 建议怎么读

先看硬件和被动顺应；再看示教与轨迹RL；对照阀门图片、臂是否移动以及真实初态。

## 此前阅读记录中的方法与实验细节

##### RBO Hand 2

- [2016 全文](https://arxiv.org/abs/1603.06348)
- [论文视频](https://youtu.be/XyZFkJWu0Q0)

七气动通道软手，物体中心示教结合轨迹 RL 和 guided policy search。阀门实际是蓝色杠杆型小气阀；回合内臂固定，主要靠不同手指推动。它证明少量驱动结合机械柔顺可以学习接触操作，但不是大半径轮辋的包络旋转。软手的被动顺应不能直接搬到电驱动 Inspire。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/valve_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
