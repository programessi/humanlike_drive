# DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning

[原文（HTML）](paper.html) · [arXiv](https://arxiv.org/abs/2410.24185) · [公开来源](https://arxiv.org/abs/2410.24185)

阅读优先级：优先阅读。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

双手动作可按并行、协作和先后依赖组织；同型号六驱动Inspire的执行与数据生成流程是有用的工程参考。

## 对当前方向盘任务的借鉴

参考基于IK的末端执行接口与承载转移的时序表示；若RL探索受阻，可将少量物理可行换手示范作为后备数据路线。

## 限制与区别

需要源示范和人工指定子任务关系；生成器不显式解决碰撞，手指主要重放源示范。它不是自主发现换手的证明。

## 建议怎么读

先看GR1/Inspire实物接口；再读三类协调关系；重点核对Appendix IX/X的人工标注、碰撞与手指动作局限。

## 此前阅读记录中的方法与实验细节

#### 3. DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning

- [论文](https://arxiv.org/abs/2410.24185)、[全文](https://arxiv.org/html/2410.24185v2)、[项目](https://dexmimicgen.github.io/)、[代码](https://github.com/NVlabs/dexmimicgen/)
- [ICRA 2025 正式会议 DOI](https://doi.org/10.1109/ICRA55743.2025.11127809)，Crossref 的 container-title 为正式 ICRA 2025 论文集；arXiv comments 也标明 ICRA 2025。

**平台尤其相关。** 真实 Fourier GR1 配两只六自由度 Inspire 手及两台 RealSense D435i。人形臂使用基于 mink 的 IK，将末端目标转为关节位置；手指直接位置控制。

**方法。** 从少量遥操作示范出发，把子任务分为 parallel、coordination、sequential：允许双臂独立执行，协作时同步相对动作，有依赖时保证顺序。经物体相对位姿变换、物理仿真执行和成功筛选扩充示范，再训练 Diffusion Policy 等模型。并不是无示范 PPO 发现换手。

**实物证据。** 罐体分类任务，从四条源示范经 real-to-sim-to-real 流程生成四十条成功的真实执行示范，训练视觉 Diffusion Policy；两类对象各十次测试，共二十次，成功率 90%，四条源示范直接训练的对照为 0%。不是“纯仿真合成数据零样本直接达到 90%”。仿真总计 21K 示范与上述真实四十条也不可混为一谈。

**可取用。** 同型号手的臂手执行接口；承载转移、释放和重新抓握的时序表示；若纯 RL 探索一直失败，可考虑少量换手示范、物理筛选扩充、再学习或微调。它没有证明无触觉 Inspire 已完成受约束连续轮盘旋转。

**全文精读补充。** Appendix IX 明确说明协作/先后关系需人工指定，生成器没有显式处理碰撞；失败轨迹筛选不能替代运行时避碰。手指动作主要重放源示范，Appendix X 指出扩充主要扩大末端动作分布，手指动作更多是局部插值。因此本项目只能借鉴其协调表示和后备数据路线，不能据此声称已经获得自主发现换手或碰撞保障。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/recent_strong_papers_review/ICRA_REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
