# CoDex: Learning Compositional Dexterous Functional Manipulation without Demonstrations

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2606.31909) · [公开来源](https://arxiv.org/abs/2606.31909)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

通过功能约束与解析可行抓握缩小RL搜索范围；无示范不等于完全无结构地逐时刻控制所有关节。

## 对当前方向盘任务的借鉴

参考初始可行抓握筛选和功能对齐；把初始化的可操作性与后续持续控制分别验收。

## 限制与区别

PPO输出动作原语参数，论文承认持续耦合臂手运动、轨迹约束和闭环交替执行仍需扩展，不是现成的连续换手策略。

## 建议怎么读

先看功能约束与抓握候选；再看38维原语参数及执行阶段；最后读真实成功定义和作者局限。

## 此前阅读记录中的方法与实验细节

#### 4. CoDex: Learning Compositional Dexterous Functional Manipulation without Demonstrations

- [ICRA 2026 官方程序](https://ras.papercept.net/conferences/conferences/ICRA26/program/ICRA26_ContentListWeb_5.html#thi1i_142)
- [正式 DOI](https://doi.org/10.1109/ICRA57385.2026.11695956)、[论文](https://arxiv.org/abs/2606.31909)、[项目](https://robin-lab.cs.utexas.edu/CoDex/)

**方法。** VLM 给出物体功能点、执行点和目标位姿等约束；解析优化生成稳定且功能对齐的抓握候选；PPO 输出 38 维动作原语参数：16 维抓握关节偏置、16 维预抓握闭合比例、6 维手位姿偏置。按预接触、抓取、闭合、移动并触发机构的顺序执行。

**实物。** Franka + LEAP，喷瓶、胶枪等六个功能操作任务，每任务五次真实测试，共三十次，整体约 73% 完成成功率。部署输入定义不含触觉，论文把加触觉列为未来方向。

**适用与限制。** 支持“用已知功能约束和可行初始抓握缩小 RL 搜索”这个设计；但不应因其 zero-demonstration 就描述成逐时刻自由关节闭环策略。作者明确指出持续耦合臂手运动、轨迹级约束、闭环反馈和交替/多点执行仍需扩展。对我们的持续换手任务，它属于辅助阅读。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/recent_strong_papers_review/ICRA_REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
