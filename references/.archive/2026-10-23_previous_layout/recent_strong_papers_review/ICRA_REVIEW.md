# ICRA 主会：对双臂双手方向盘操作有用的参考

核验日期：2026-10-22。本次读取 ICRA 2026 官方 PaperCept 主会程序及公开论文方法/实验；ICRA 2025 的两篇通过 Crossref 正式会议 DOI 核实。没有运行论文代码、训练新策略或操作实物。

## 阅读优先级

1. **DexCtrl，ICRA 2026**：无触觉条件下，用本体与控制历史适应底层控制差异。最对应初始接触受力难复制的担忧。
2. **DexScrew，ICRA 2026**：固定轴几何简化与自主指步态，最对应受约束连续转动；但完整实物方法用真实触觉数据。
3. **DexMimicGen，ICRA 2025**：真实双六自由度 Inspire 手、双臂协同时序与数据生成；属于示范驱动路线。
4. **CoDex，ICRA 2026**：了解如何用功能约束与可行抓握缩小探索，但不是持续闭环换手控制的主要参考。

这几篇分别覆盖迁移、受约束旋转和双手协调。不能把它们合并描述成已经证明了“无触觉 Inspire 双手长时精确轮角跟踪与自主换手”。

## 1. DexCtrl: Sim-To-Real Dexterity with Adaptive Controller Learning

- [ICRA 2026 官方主会程序条目](https://ras.papercept.net/conferences/conferences/ICRA26/program/ICRA26_ContentListWeb_4.html#wei1i_213)
- [公开论文](https://arxiv.org/abs/2505.00991)、[HTML 全文](https://arxiv.org/html/2505.00991)
- 公开版本标题带 **Towards**：*DexCtrl: Towards Sim-to-Real Dexterity with Adaptive Controller Learning*。下列方法与数字来自 2025-05-02 v1；主会身份由 2026 官方程序独立确认，不将旧预印本日期当作会议日期。

**任务。** 16-DoF LEAP 单手指内转动和借桌面翻转。关心相同关节目标轨迹在不同底层增益下产生不同接触力，造成迁移失败。

**训练。** PPO oracle 同时输出关节目标增量与 PD 增益。Oracle 输入包含三帧关节实测位置、目标位置、控制参数，以及特权物体位姿和物理属性。收集仿真轨迹后，分别训练动作预测模块和控制参数预测模块：前者用 self-attention 编码历史，后者以当前预测动作作 query、以历史作 key/value 的 cross-attention 预测增益。

**部署。** 使用最近十步的关节实测位置、目标位置与控制参数历史，闭环预测关节目标及 PD 参数。其部署定义不要求触觉或接触力输入。原文通过近似增益上下界将仿真控制参数映射到实物，不等于完全无需硬件校准。

**实物证据。** 论文称测试十二个未见物体、每物体十个随机初始姿态。正文 Table 3 所列五种物体的平均净旋转：固定增益手调基线 2.431 rad、去掉自适应 PD 的方法 4.986 rad、完整方法 11.041 rad；这是回合净转角，不是精确目标跟踪成功率，也不是圈数。TTF 上限 300 的时间单位不要自行解释成秒。

**对我们有用。** 将目标与实际运动的偏差、过去命令及执行配置纳入交互历史；将执行器/控制差异作为迁移的核心变量，而不是只随机化摩擦和物体质量。初始受力不必逐点一致，但策略必须训练于有效接触分布，并从可获得反馈适应。

**不能直接照搬。** 论文假设可调 PD/关节力矩执行。Inspire 是否能在线调相应底层参数尚未核实；电流可读也不等于 PD 可写。若只有厂商封装的位置接口，只能先借鉴历史条件策略、执行器建模和随机化，不能声称已经实现 DexCtrl。单手自由物体转动也未证明双臂固定轴轮盘的换手与避碰。

## 2. DexScrew / Learning Dexterous Manipulation Skills from Imperfect Simulations

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

## 3. DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning

- [论文](https://arxiv.org/abs/2410.24185)、[全文](https://arxiv.org/html/2410.24185v2)、[项目](https://dexmimicgen.github.io/)、[代码](https://github.com/NVlabs/dexmimicgen/)
- [ICRA 2025 正式会议 DOI](https://doi.org/10.1109/ICRA55743.2025.11127809)，Crossref 的 container-title 为正式 ICRA 2025 论文集；arXiv comments 也标明 ICRA 2025。

**平台尤其相关。** 真实 Fourier GR1 配两只六自由度 Inspire 手及两台 RealSense D435i。人形臂使用基于 mink 的 IK，将末端目标转为关节位置；手指直接位置控制。

**方法。** 从少量遥操作示范出发，把子任务分为 parallel、coordination、sequential：允许双臂独立执行，协作时同步相对动作，有依赖时保证顺序。经物体相对位姿变换、物理仿真执行和成功筛选扩充示范，再训练 Diffusion Policy 等模型。并不是无示范 PPO 发现换手。

**实物证据。** 罐体分类任务，从四条源示范经 real-to-sim-to-real 流程生成四十条成功的真实执行示范，训练视觉 Diffusion Policy；两类对象各十次测试，共二十次，成功率 90%，四条源示范直接训练的对照为 0%。不是“纯仿真合成数据零样本直接达到 90%”。仿真总计 21K 示范与上述真实四十条也不可混为一谈。

**可取用。** 同型号手的臂手执行接口；承载转移、释放和重新抓握的时序表示；若纯 RL 探索一直失败，可考虑少量换手示范、物理筛选扩充、再学习或微调。它没有证明无触觉 Inspire 已完成受约束连续轮盘旋转。

**全文精读补充。** Appendix IX 明确说明协作/先后关系需人工指定，生成器没有显式处理碰撞；失败轨迹筛选不能替代运行时避碰。手指动作主要重放源示范，Appendix X 指出扩充主要扩大末端动作分布，手指动作更多是局部插值。因此本项目只能借鉴其协调表示和后备数据路线，不能据此声称已经获得自主发现换手或碰撞保障。

## 4. CoDex: Learning Compositional Dexterous Functional Manipulation without Demonstrations

- [ICRA 2026 官方程序](https://ras.papercept.net/conferences/conferences/ICRA26/program/ICRA26_ContentListWeb_5.html#thi1i_142)
- [正式 DOI](https://doi.org/10.1109/ICRA57385.2026.11695956)、[论文](https://arxiv.org/abs/2606.31909)、[项目](https://robin-lab.cs.utexas.edu/CoDex/)

**方法。** VLM 给出物体功能点、执行点和目标位姿等约束；解析优化生成稳定且功能对齐的抓握候选；PPO 输出 38 维动作原语参数：16 维抓握关节偏置、16 维预抓握闭合比例、6 维手位姿偏置。按预接触、抓取、闭合、移动并触发机构的顺序执行。

**实物。** Franka + LEAP，喷瓶、胶枪等六个功能操作任务，每任务五次真实测试，共三十次，整体约 73% 完成成功率。部署输入定义不含触觉，论文把加触觉列为未来方向。

**适用与限制。** 支持“用已知功能约束和可行初始抓握缩小 RL 搜索”这个设计；但不应因其 zero-demonstration 就描述成逐时刻自由关节闭环策略。作者明确指出持续耦合臂手运动、轨迹级约束、闭环反馈和交替/多点执行仍需扩展。对我们的持续换手任务，它属于辅助阅读。

## 另外两篇已核实但优先级较低

- **DextrAH-RGB: Visuomotor Policies to Grasp Anything with Dexterous Hands，ICRA 2026**：[官方程序](https://ras.papercept.net/conferences/conferences/ICRA26/program/ICRA26_ContentListWeb_3.html#tui2i_141)、[论文](https://arxiv.org/abs/2412.01791)。本次核实主会摘要和 arXiv 摘要，未精读全文。仿真 RL + geometric fabric controller + RGB 学生蒸馏，主要解决抓取和视觉迁移。可为臂手统一控制的结构化底层提供参考，但不是连续换抓的直接实验证据。其预印本始于 2024 年，不能据此误写为 ICRA 2025。
- **Learning Visuotactile Skills with Two Multifingered Hands / HATO，ICRA 2025**：[论文](https://arxiv.org/abs/2404.16823)、[项目](https://toruowo.github.io/hato/)、[正式 DOI](https://doi.org/10.1109/ICRA55743.2025.11128180)。本次核实会议记录、摘要与项目页。双臂多指手遥操作与视触觉模仿学习，适合看双手数据采集和传感消融；不是当前无触觉纯仿真 RL 的主要依据。

## 对当前路线的实际含义

保留上游给目标轮角/速度参考的任务边界。先学可达的小角度闭环执行和有效接触恢复，再拓展承载转移、单手释放/重新布置和反向；不要把固定腕或固定抓握正则当作所有阶段的永久目标。部署 actor 只使用可获取信息，训练可用特权 critic 或 teacher。

迁移优先核实真实执行器接口，并让训练覆盖初始位置偏差、闭合量、摩擦、延迟、死区、执行速度/力限制等可实现分布。这里的具体轮盘课程和指标是对本任务的设计建议，不是上述论文已经验证的结论。

原始公开来源与下载记录在本目录被忽略的 `raw/`。其中 `icra26_day3/4/5.html` 为正式主会程序，`icra26_selected_abstracts.json` 保存筛选摘要，`sources.jsonl` 保存访问 URL、状态和哈希。DBLP 返回机器人验证页，未用作发表证据；部分 Crossref 批量查询返回 429，未把失败查询当作否定发表的证据。
