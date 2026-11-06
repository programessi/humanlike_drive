# IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2305.17103) · [公开来源](https://www.roboticsproceedings.org/rss19/p039.html)

阅读优先级：优先阅读。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

策略可能利用不真实穿透获得奖励；任务空间执行与增量积分也会改变稳态误差和接触下的累积行为。

## 对当前方向盘任务的借鉴

审查数值接触漏洞、初态课程和部署动作语义；结合我们腕部持续漂移，重点检查积分命令与实际执行误差。

## 限制与区别

装配环境与双手轮盘不同；目标积分可能windup，需要抑制。作者未采用动力学随机化，不能照搬为所有任务通用配置。

## 建议怎么读

先看SAPU与穿透处理；再看SDF奖励和采样课程；重点读部署控制、Policy-Level Action Integrator与抗windup说明。

## 此前阅读记录中的方法与实验细节

#### 3. IndustReal：接触真实性和执行控制共同决定精度

[RSS官方全文](https://www.roboticsproceedings.org/rss19/p039.pdf)，[DOI](https://doi.org/10.15607/rss.2023.xix.039)。读取正文及实物系统附录。

- Simulation-Aware Policy Update检查穿透并过滤/加权训练回合，降低策略利用数值接触漏洞完成任务的可能。
- SDF奖励量化几何对齐；采样课程让策略覆盖不同插入深度，避免只在已经部分插入的状态成功。
- actor使用关节角、夹爪/物体/目标位姿；因实物速度和力矩噪声较大，将额外信息保留给critic。
- 使用Franka可执行的任务空间阻抗控制，而非依赖更精确动力学模型的OSC。
- 随机化初态与观测；作者有强动力学先验，明确没有采用动力学随机化。说明没有所有任务通用的随机化清单。
- Policy-Level Action Integrator将增量加到上一个目标，而非实测状态，减少稳态偏差；论文明确指出接触时会windup，并提出抑制方案。不能直接将无界积分搬到方向盘控制。
- 感知模块还使用真实图像微调；“不做真实策略适配”不等于整个系统没有真实数据。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/manipulation_sim2real_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
