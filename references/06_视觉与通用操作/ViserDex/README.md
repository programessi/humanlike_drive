# ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation

[原文（PDF）](paper.pdf) · [arXiv标题检索（编号未确认）](https://arxiv.org/search/?query=ViserDex%3A+Visual+Sim-to-Real+for+Robust+Dexterous+In-hand+Reorientation&searchtype=title&abstracts=show&order=-announced_date_first&size=50) · [公开来源](https://www.roboticsproceedings.org/rss22/p150.html)

阅读优先级：专题补充。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

操作教师、带记忆学生和视觉估计分工明确；学生不仅模仿动作，还学习表示不可直接观测的状态。

## 对当前方向盘任务的借鉴

当前可借教师学生、belief重建和按表现推进课程；视觉部分等接触执行稳定后再接入。

## 限制与区别

单手重定向不是双臂换手；3DGS视觉链路和较大网络并不自动解决当前关节漂移或接触课程问题。

## 建议怎么读

先画teacher/student观测表；再读belief encoder、重建和BC损失；最后看课程推进、真实单目估计与消融。

## 此前阅读记录中的方法与实验细节

#### 3. ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation

- [RSS 官方论文页](https://www.roboticsproceedings.org/rss22/p150.html)、[正式全文](https://www.roboticsproceedings.org/rss22/p150.pdf)。
- 作者：Arjun Bhardwaj、Maximum Wilder-Smith、Mayank Mittal、Vaishakh Patil、Marco Hutter。
- 发表依据：RSS XXII 官方论文集，2026 年 7 月，DOI `10.15607/RSS.2026.XXII.150`。

**三部分分工。** 先用特权状态训练 RL teacher 学接触操作，再蒸馏到 recurrent student，最后用单目 RGB 的物体位姿估计部署。Student 的 belief encoder 从带噪本体与感知观测更新隐状态；训练 decoder 重建特权/外部状态，同时以行为克隆拟合 teacher 动作。真机不向 student 提供指尖接触力；接触力属于 teacher 的特权信息。视觉部分用 3D Gaussian Splatting 与渲染前随机化缩小视觉 gap。

**课程。** 随任务成功表现逐步提高动作正则、延迟与完成效率要求，避免从开始就以强正则抑制探索；不是固定训练若干步后无条件升级。

**证据与边界。** 正文报告真实多指手与单 RGB 相机上五种物体的重定向。本文不是双手换握任务。对当前无视觉阶段，值得吸收的是 teacher/student、belief 重建和按表现推进的课程；3DGS 视觉链路暂不需要搬入。

它提供了比“直接给 actor 加 GRU”更完整的部分观测学习方案，但仍需在本任务中做消融验证，并不保证 teacher 的动作在 Inspire 的执行误差下保持同样效果。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/recent_strong_papers_review/REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
