# Learning Dexterous Manipulation Skills from Imperfect Simulations

[原文（PDF）](paper.pdf) · [arXiv](https://arxiv.org/abs/2512.02011) · [公开来源](https://arxiv.org/abs/2512.02011)

阅读优先级：优先阅读。整理日期：2026-10-23。

## 阅读范围

已有论文全文；以下按此前方法与实验相关阅读记录整理，本次没有新增实验复现。

## 我的理解

固定轴简化模型可以先学出旋转和指步态，不必精确复刻螺纹；但基础旋转技能与完整真实紧固任务是两种能力。

## 对当前方向盘任务的借鉴

借用简化固定轴任务、有效接触初态和教师到历史学生的训练分工；为必要的局部释放保留动作空间。

## 限制与区别

最终完整实物路线包含技能辅助遥操作、真实示范和触觉BC。不能用最终视频证明我们无触觉、纯仿真训练就能直接迁移。

## 建议怎么读

依次看简化仿真与PPO动作、oracle/历史学生、真实数据采集流程；对照Table II与IV-E，区分任务进度和完整成功率。

## 此前阅读记录中的方法与实验细节

#### 2. DexScrew / Learning Dexterous Manipulation Skills from Imperfect Simulations

- [ICRA 2026 官方主会程序条目](https://ras.papercept.net/conferences/conferences/ICRA26/program/ICRA26_ContentListWeb_5.html#thi2i_276)
- [论文](https://arxiv.org/abs/2512.02011)、[全文](https://arxiv.org/html/2512.02011)、[项目](https://dexscrew.github.io/)
- arXiv v2 的 Journal reference 同样写明 ICRA 2026。它是主会论文，不是仅凭 workshop 页面作判断。

**三段路线。**

1. 将螺母或手柄建模为通过 revolute joint 连接固定基座的简化物体，在仿真中训练 PPO 学转动与指步态；不建精确螺纹。
2. 先训练特权 oracle，再以本体历史预测隐变量并蒸馏 sensorimotor primitive。手指动作是目标位置增量，再由低层 PD 执行。
3. 用该技能辅助真实遥操作：人控制臂，触发学到的手指旋转技能，采集真实触觉和本体数据；最后行为克隆学习协调臂手的多模态策略。

**可取用。** “固定轴 + 简化转动惯量/摩擦”的模型完全可以是学习接触重排的起点；不用先获得精确螺纹/轮盘接触的数字孪生。应使转动奖励与动作约束允许局部释放和重新接触，避免把初始抓握永久固定。

**边界。** 文中明确说简化仿真策略可以产生合理旋转，但不能直接完成完整真实任务。最终效果依赖真实示范和触觉；没有证明本体历史能完全替代触觉。评价中存在 task progress ratio，不能将进度百分比当作完整成功率。我们可以借第一段课程；当前硬件不能直接照搬最后一段。

**全文精读补充。** 上述“依赖触觉”指最佳完整配置，不应理解成作者没有验证无触觉版本。Table II 中，无触觉但带历史的真实数据 BC 策略，螺丝刀 progress ratio 为 87.50±18.61%，触觉加历史为 95.00±13.24%；直接 sim2real 为 41.60±26.21%。这支持历史的价值，但无触觉 BC 同样用过真实示范，且 progress ratio 不是任务完成成功率。§IV-E 还比较了特权教师、历史学生、仅 critic 特权的 PPO 和无特权 PPO；该任务上教师/学生更好，不能推广为所有任务都必须蒸馏。正文与附录叙述训练 1.5e9 步，Table VI 却列 3e9，引用训练预算时应保留这个不一致。

## 记录来源

以上整理自[此前专题阅读记录](../../.archive/2026-10-23_previous_layout/recent_strong_papers_review/ICRA_REVIEW.md)；文中论文结果不代表本项目已经达到相同能力。
