# AsymDex: Asymmetry and Relative Coordinates for RL-based Bimanual Dexterity

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2411.13020) · [公开来源](https://arxiv.org/abs/2411.13020)

阅读优先级：优先阅读。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

相对坐标和主辅手分工降低双手协调难度，说明把臂冗余交给执行层并不排斥学习接触协同。

## 对当前方向盘任务的借鉴

参考相对腕部表达和分层执行；方向盘任务可共享协调状态，但必须允许左右承载角色随构型与目标方向变化。

## 限制与区别

固定主辅关系不等于自主交替换手。此前重新核对的发表信息是WCBM@RSS workshop展示，不能写成RSS主会。

## 建议怎么读

先看主辅手与相对坐标定义；再对照仿真与hardware配置的观测/动作；最后读实物初始抓握和角色限制。

## 此前阅读记录中的方法与实验细节

#### 4. AsymDex：与你提出的臂手分层思路最接近

**AsymDex: Asymmetry and Relative Coordinates for RL-based Bimanual Dexterity**，Zhaodong Yang、Yunhai Han、Ai-Ping Hu、Harish Ravichandar。初稿 2024-11，读取的 v3 更新于 2025-06-28。

- [论文](https://arxiv.org/abs/2411.13020)
- [项目](https://sites.google.com/view/asymdex-2025/)
- [官方代码](https://github.com/GT-STAR-Lab/AsymDex)

主要研究 **已经抓住之后的 interaction phase**；接近抓取另作 acquisition phase。策略不依赖人类动作示教。

它给两手不同角色：

- 主手学复杂手指运动和相对腕部运动。
- 辅助手保持固定手指抓握，只调整物体/手基座位置和朝向，不让它学习不必要的指内运动。
- 观测使用主手关节状态、主手腕相对所持物体的位姿、被操作物体相对位姿。
- 动作使用主手手指目标和相对腕位姿；相对位姿控制器把任务分配给两个腕部，再由基础控制执行。

真实任务设置中，论文给出 30 维观测、22 维动作（16 个主手指关节 + 六维相对基座目标）。PPO actor 为 `[256,256,128]` ELU MLP，value 为 `[512,512,512]`。

真实硬件是 Allegro 16 自由度操作手和 Ability 六自由度辅助手，各装在七自由度 Kinova Gen3 上；不是两只同型号手。瓶盖任务实物成功 18/20 次。仿真 twist-lid 成功阈值是转过 3π rad，并非精确角度跟踪评价。

核验的硬件任务源码使用 `_dls_ik`，说明该公开实现采用阻尼最小二乘 IK，**不是已经验证了 QP 优化或人体姿态目标**。你可以另选 QP 作为下层，但应区分论文已有控制与拟议替换。

方向盘两手作用在同一个刚性转动件上，不能机械复制瓶身/瓶盖的“保持件—操作件”分工；可借鉴相对坐标与任务角色，重新定义共同转动和可切换的承载关系。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/bimanual_rl_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
