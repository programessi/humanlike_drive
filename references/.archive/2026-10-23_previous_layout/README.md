# 双臂双手方向盘操作：论文与阅读笔记

这里集中保存此前讨论、检索和阅读过的资料。2026-10-23 从 `manipulation/references/` 整体迁到此处，现有 **30篇不同论文的PDF、2篇补充HTML全文** 已建立便于查找的入口；其余讨论过的工作保留在线链接与阅读范围说明。

- [完整论文索引](PAPER_INDEX.md)：本地全文、论文标题、公开来源、与当前任务的关系和阅读笔记。
- [PDF文件夹](pdfs/)：按论文简称命名，可直接打开阅读。
- [HTML全文文件夹](fulltexts/)：DexMimicGen、TRANSIC；缓存页面的图片等外部资源不保证离线可用。
- [结构化目录](CATALOG.json)：32条全文入口及其来源路径。

论文索引优先列arXiv：目前29篇全文条目已有来自既有资料的直接编号链接；ViserDex、DexImit和Extrinsic Dexterity的编号尚未确认，列出完整标题和arXiv标题检索入口。未下载全文的工作也保留标题和已知链接；没有把“检索入口”写成已经确认的论文编号。

## 建议先读

围绕当前条件：H1-2双臂、双六驱动Inspire手、无触觉、已知初始抓握和轮盘几何、跟踪带时间要求的轮角参考，逐步学习接管与换手。

| 顺序 | 论文与全文 | 重点看什么 | 阅读笔记 |
| --- | --- | --- | --- |
| 1 | [DexCtrl](pdfs/DexCtrl.pdf) | 命令与实际执行历史、控制器差异、初始接触难复制时如何适应 | [ICRA精读](recent_strong_papers_review/ICRA_REVIEW.md) |
| 2 | [DexScrew](pdfs/DexScrew.pdf) | 固定轴简化模型、持续旋转与指步态、特权教师到历史学生 | [ICRA精读](recent_strong_papers_review/ICRA_REVIEW.md) |
| 3 | [HORA](pdfs/HORA.pdf) | 不用触觉的本体历史适应与持续接触操作 | [接触切换](contact_switching_review/REVIEW.md) |
| 4 | [Twisting Lids](pdfs/Twisting_Lids.pdf) | 无示范的双手旋拧RL；同时注意固定腕部的任务边界 | [双手旋拧](bimanual_rl_review/REVIEW.md) |
| 5 | [Grasp2Twist](pdfs/Grasp2Twist.pdf) | 统一臂手策略、抓握奖励、局部释放和课程；此前记录为预印本 | [双手旋拧](bimanual_rl_review/REVIEW.md) |
| 6 | [DexMimicGen（HTML）](fulltexts/DexMimicGen.html) | 双Inspire执行接口、双手协调与先后关系；这是示范驱动方法 | [ICRA精读](recent_strong_papers_review/ICRA_REVIEW.md) |
| 7 | [CoDex](pdfs/CoDex.pdf) | 用功能约束和可行初始抓握缩小探索 | [ICRA精读](recent_strong_papers_review/ICRA_REVIEW.md) |
| 8 | [SimToolReal](pdfs/SimToolReal.pdf) | 物体中心表示、工具操作与迁移设计 | [近期主会工作](recent_strong_papers_review/REVIEW.md) |

这些论文分别解决不同问题。固定轴转动、双手旋拧、本体历史和迁移可以借鉴，但它们并未共同证明本项目已经能够自主换手。尤其要区分无触觉部署、纯仿真训练、真实数据微调与最终完整实物任务的结果。会议身份、版本日期及方法限制详见原笔记；本次整理没有重新核验网络来源。

## 按问题查资料

| 主题 | 笔记入口 | 主要工作或资料 |
| --- | --- | --- |
| ICRA、当前技术路线依据 | [ICRA_REVIEW](recent_strong_papers_review/ICRA_REVIEW.md) | DexCtrl、DexScrew、DexMimicGen、CoDex；补充HATO与DextrAH-RGB |
| 新近主会操作论文 | [近期论文综述](recent_strong_papers_review/REVIEW.md) | SimToolReal、DexterityGen、ViserDex、DexImit等 |
| 双臂双手旋拧RL | [双手旋拧综述](bimanual_rl_review/REVIEW.md) | Grasp2Twist、Twisting Lids、AsymDex、Inspire工具操作、CHORD等 |
| 连续转动、接触切换与换握 | [接触切换综述](contact_switching_review/REVIEW.md) | HORA、DexScrew、DexMani、PatternDex、Proprioceptive Transformer等 |
| 仿真训练到实物部署 | [sim2real综述](manipulation_sim2real_review/REVIEW.md) | Dactyl、魔方、DeXtreme、IndustReal、TRANSIC；初始接触与执行接口 |
| 阀门任务与六驱动手硬件 | [阀门与Inspire综述](valve_review/REVIEW.md) | COMAN、DexArt、CyberDemo、DIPS、DClaw、RH56DFX标定 |
| 人类驾驶握持与换手 | [驾驶行为笔记](human_steering_review/REVIEW.md) | 握持位置、低速转向、双手协调论文及驾驶教学依据 |
| 早期方向盘相关工作 | [RELATED_WORK](RELATED_WORK.md) | Musashi实车驾驶、方向盘导纳控制及早期基线范式说明 |

## 文件布局与来源

```text
references/
  README.md                       阅读入口
  PAPER_INDEX.md                  全文目录与补充在线文献
  CATALOG.json                    机器可读目录
  pdfs/                           30篇PDF的简明文件名入口
  fulltexts/                      2篇HTML全文入口
  recent_strong_papers_review/     ICRA/RSS笔记及原始来源
  bimanual_rl_review/              双手RL笔记、论文、公开代码摘录
  contact_switching_review/        连续旋转与换握笔记、原始来源
  manipulation_sim2real_review/    迁移笔记与论文
  valve_review/                   阀门、Inspire硬件与代码摘录
  human_steering_review/           人类驾驶行为笔记
  RELATED_WORK.md                  初步文献核对
  RELOCATION.json                 本次迁移的原文件清单与SHA256
```

`pdfs/`、`fulltexts/` 内是相对符号链接，指向本文件夹内保留的原始文件，不重复复制大文件。整个 `references/` 文件夹一起复制即可保留这些关系。

原有260个文件均保留并核对SHA256，包括PDF、提取文本、HTML、来源记录、检索响应和代码摘录。`dactyl_retry.pdf` 是原先的重复下载，保留原文件，阅读目录只列一次。原始下载记录中的旧绝对路径保留为历史记录，当前入口以本目录索引为准。部分历史笔记描述的是当时的基线或方案，当前实现状态请查看[第一阶段实验记录](../manipulation/results/regrasp_stage1/README.md)。
