# 新近主会论文：目标跟踪、接触重排与实物迁移

核验日期：2026-10-22。按用户要求，补充近期发表依据更明确的方法参考。本次从 RSS 2025/2026 官方论文集核验主会身份，阅读下列论文的方法与实物实验相关内容；不是按发表场所给方法质量作绝对排名，也不是完全对应方向盘换手的论文清单。没有启动训练或实物操作。

## 建议阅读顺序

1. **SimToolReal，RSS 2026**：最贴近“给定物体运动目标，学习腕手如何执行”的问题定义。
2. **DexterityGen / DexGen，RSS 2025**：理解如何把保持抓握与抓握间转移学成可复用的底层能力。
3. **ViserDex，RSS 2026**：理解特权教师、部分观测学生与有记忆的部署策略如何分工。
4. **DexImit，RSS 2026**：未来引入拟人双手行为参考时的补充，不作为当前纯 RL 接触重排的主要证据。

## 1. SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation

- [RSS 官方论文页](https://www.roboticsproceedings.org/rss22/p151.html)、[正式全文](https://www.roboticsproceedings.org/rss22/p151.pdf)、[项目](https://simtoolreal.github.io/)。
- 作者：Kushal Kedia、Tyler Ga Wei Lum、Jeannette Bohg、Karen Liu。
- 发表依据：RSS XXII 官方论文集，2026 年 7 月，DOI `10.15607/RSS.2026.XXII.151`。

**任务与方法。** 在程序生成的工具几何上训练一个目标位姿条件策略。训练中随机生成当前及后续目标位姿；部署时给定物体目标位姿序列，由机器人闭环执行。输入为本体状态、当前物体 6D 位姿、固定的粗抓握区域包围盒及当前目标位姿。使用 LSTM 处理交互历史，输出 29 个臂手关节的位置目标（7-DoF KUKA + 22-DoF Sharpa）。部署输入定义不要求触觉阵列或接触力。

训练使用 **SAPG（PPO 的种群探索变体）**，不是仅扩大普通 PPO 网络。作者指出标准 PPO 在大规模并行仿真中仍会遭遇探索瓶颈。使用特权 critic，并随机化延迟、感知与执行误差、抓握区域误差和物体外力/力矩扰动。目标奖励只奖励相对历史最小距离的进一步进展，并配合到达奖励，避免仅停在目标附近持续获取稠密奖励。

**与本任务特别相关的实验。** 固定抓握对照在需要工具指内转动时，无法仅靠机械臂姿态完成动作，甚至出现机械臂与桌面碰撞；策略通过臂手协调和指内旋转继续执行。正文 Fig.5 在需要旋转的刷子任务中报告 Task Progress：方法 82.7%、固定抓握 10.8%。这是该具体任务的目标进度，不是通用成功率。

**证据与边界。** 有 120 次真实执行，覆盖 24 个任务、12 个对象、6 类工具。评价重点是目标序列进度，目标容差以物体关键点位置距离 2 cm 定义；作者明确不等同于实际功能任务完成。序列是**达到当前目标后再切换**，没有证明严格按时间跟踪。是单臂单手的自由工具操作，未证明固定轴方向盘的双手交替接管；也不能把它的厘米容差换算为我们已具备 1°轮角精度。运行时物体目标来自人类 RGB-D 视频，但 RL 不使用人类手指动作示范逐关节模仿。

**可取用。** 把物体目标与机器人实现动作分开；利用历史适应交互；正视探索瓶颈；针对固定抓握的局限设计对照。我们的轮角和带时间序列可以替代物体 6D 目标，但固定轴约束、双手承载与精确停留需要另行设计。

## 2. DexterityGen: Foundation Controller for Unprecedented Dexterity

- [RSS 官方论文页](https://www.roboticsproceedings.org/rss21/p103.html)、[正式全文](https://www.roboticsproceedings.org/rss21/p103.pdf)。
- 发表依据：RSS XXI 官方论文集，2025 年 6 月，DOI `10.15607/RSS.2025.XXI.103`。
- 正式 PDF 的文本层为空，本次渲染并目视阅读正文第 3–8 页的方法、系统与实验；不将空文本提取当作已读全文。

**关键思路。** 先在仿真生成有效抓握集合，以 **Anygrasp-to-Anygrasp** 为中心任务训练 RL：从一个有效抓握出发，目标是附近可达的另一个抓握，达到后继续更新目标。原文强调直接要求远距离抓握转移很困难，附近目标的选择很重要。同时改变物体、物理属性和腕部姿态/运动。

然后用 RL 生成的数据训练生成式底层控制器：UNet diffusion 生成未来手指关键点运动，残差 MLP 逆动力学模型把运动转换为关节位置目标。控制模型使用本体历史，外部运动指令通过采样引导作用于生成动作；真实实验中人类通过手套/Vive 提供高层指令，机械臂的腕部运动另行控制。

**对我们有用。** 抓握不应被视为一个永远不变的闭合姿态；可以先学习有效抓握之间的可达转移，再用目标任务组织连续动作。多模态运动先验也可避免不同重抓方式被平均，但这不意味着必须立即给当前模型增加 diffusion。

**重要边界。** 这是单 Allegro 手的生成式辅助控制，不是完全自主的双臂角色切换。实物长任务依赖人类引导，精密旋拧还使用模式条件；不能描述为纯 PPO 自主解决完整工具使用。正文 Table II 螺丝刀完整使用为 3/20、注射器为 4/20，不能仅用亮眼视频代表可靠性。数据与训练规模很大：约 `10^10` transitions；生成模型和逆动力学模型训练使用 96 GPU、约 3 天。当前工作站适合借鉴课程和任务分解，不能直接承诺完整复现同规模基础控制器。

## 3. ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation

- [RSS 官方论文页](https://www.roboticsproceedings.org/rss22/p150.html)、[正式全文](https://www.roboticsproceedings.org/rss22/p150.pdf)。
- 作者：Arjun Bhardwaj、Maximum Wilder-Smith、Mayank Mittal、Vaishakh Patil、Marco Hutter。
- 发表依据：RSS XXII 官方论文集，2026 年 7 月，DOI `10.15607/RSS.2026.XXII.150`。

**三部分分工。** 先用特权状态训练 RL teacher 学接触操作，再蒸馏到 recurrent student，最后用单目 RGB 的物体位姿估计部署。Student 的 belief encoder 从带噪本体与感知观测更新隐状态；训练 decoder 重建特权/外部状态，同时以行为克隆拟合 teacher 动作。真机不向 student 提供指尖接触力；接触力属于 teacher 的特权信息。视觉部分用 3D Gaussian Splatting 与渲染前随机化缩小视觉 gap。

**课程。** 随任务成功表现逐步提高动作正则、延迟与完成效率要求，避免从开始就以强正则抑制探索；不是固定训练若干步后无条件升级。

**证据与边界。** 正文报告真实多指手与单 RGB 相机上五种物体的重定向。本文不是双手换握任务。对当前无视觉阶段，值得吸收的是 teacher/student、belief 重建和按表现推进的课程；3DGS 视觉链路暂不需要搬入。

它提供了比“直接给 actor 加 GRU”更完整的部分观测学习方案，但仍需在本任务中做消融验证，并不保证 teacher 的动作在 Inspire 的执行误差下保持同样效果。

## 4. DexImit: Learning Bimanual Dexterous Manipulation from Monocular Human Videos

- [RSS 官方论文页](https://www.roboticsproceedings.org/rss22/p003.html)、[正式全文](https://www.roboticsproceedings.org/rss22/p003.pdf)。
- 发表依据：RSS XXII 官方论文集，2026 年 7 月，DOI `10.15607/RSS.2026.XXII.003`。

从人类视频重建手物关系，分解子任务并进行双手调度，合成满足物理约束的机器人轨迹，数据增强后训练 3D Diffusion Policy。真实平台为双 UR5e + XHand + Azure Kinect，评价单手、独立双手、双手抓同一物体、协作倒水等任务。

适合参考未来拟人行为的数据表达和双手时序，但**不是无示范 RL 自主发现重抓**。正文局限明确指出当前不能处理复杂指内操作，且关节物体适配仍属后续工作，故不选作当前方向盘接触重排的主方法。

## 补充候选与排除理由

- **Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning，RSS 2026**：[官方页](https://www.roboticsproceedings.org/rss22/p149.html)。阅读了方法与实物相关章节；使用接触诱导动力学表示/world model 条件化 RL，有真实杂乱场景实验。核心是借环境实现非抓持操作，不是双灵巧手换握，相关度低于前三篇。
- **One Hand to Rule Them All，RSS 2026**：[官方页](https://www.roboticsproceedings.org/rss22/p124.html)。本次核验官方摘要和发表记录，重点是跨手型表示与抓握迁移，不把它当作连续旋转或换手能力证据。
- **ArticuBot，RSS 2025**：[官方页](https://www.roboticsproceedings.org/rss21/p156.html)。本次核验官方摘要和发表记录，关节物体 sim2real 很有价值，但主要是采样/规划生成示范再蒸馏，不是灵巧手接触重排 RL。
- **DexNDM**：[arXiv](https://arxiv.org/abs/2510.08556)。摘要显示通过真实数据学习关节神经动力学进行仿真策略动作适配，值得后续迁移阶段阅读；本次没有核实主会/期刊录用，因此不归入已核验主会主推荐。
- **AsymDex**：本次重新读取 [arXiv](https://arxiv.org/abs/2411.13020)，Comments 为 `WCBM @ RSS 2025 Spotlight Presentation`，属于 workshop 展示信息；不能写成 RSS 主会录用。
- **Grasp2Twist**：此前核实的 arXiv 为 `Under review`，继续作为任务相关预印本补充，不承担已发表主会证据的角色。
- **Twisting Lids Off with Two Hands，CoRL 2024**：[论文](https://arxiv.org/abs/2403.02338)。虽较早，仍是无示范、无接触力 actor 的双手旋拧直接参考，值得和新论文一起阅读。

## 对本项目路线的影响

这批论文没有共同证明“换大网络即可学会换手”。更直接的共同点是：定义可实现的目标条件任务，建立有效抓握/转移分布，处理探索和部分观测，并明确真实执行接口与评价口径。

下一轮设计优先参考 SimToolReal 的目标条件臂手执行和 DexGen 的附近抓握转移课程；如果直接部分观测 RL 始终学不好，再比较 ViserDex 式 teacher/student，而不是未经实验同时加入生成模型、world model 和新动作接口。我们的时限轮角跟踪、双手角色切换和六驱动手能力仍是需独立验证的内容。

本次没有修改现有训练/控制代码，也没有以新的文献取代实验。原始官方网页、正式 PDF 与来源/哈希清单保存在被忽略的 `raw/`。普通搜索接口返回过不相关结果，未将其用于方法或发表状态判断；DexGen 猜测项目地址返回 404，推荐链接使用已核验的 RSS 正式论文页。
