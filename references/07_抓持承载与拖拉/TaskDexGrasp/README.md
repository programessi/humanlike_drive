# Task-Oriented Dexterous Hand Pose Synthesis Using Differentiable Grasp Wrench Boundary Estimator

[原文](paper.pdf) · [DOI](https://doi.org/10.1109/IROS58592.2024.10802652) · [论文给出的项目](https://pku-epic.github.io/TaskDexGrasp/)

Jiayi Chen、Yuxing Chen、Jialiang Zhang、He Wang，IROS 2024。原文来自本地Zotero；2026-10-26核对任务wrench、姿态优化、真实实验及限制，没有复现。

## 方法

TWS表示任务所需的力/力矩集合，GWS表示当前接触可提供的集合。通过可微分GWS边界估计与梯度优化，让手位姿支持指定任务，而不要求任何时候都抵抗任意方向外载。任务包括转动、抬起、推拉和精细抓持。

真实平台LEAP加UR5e，使用扫描物体网格与ICP估计位姿。运动规划到达合成抓姿后，**手指关节保持固定，后续手根运动人工设计**。真实拧盖、转把手、拧旋钮、拉抽屉各10次，成功分别5/10、2/10、6/10、8/10。转把手低成功率与较大回正力有关。

## 对方向盘的取用

适合生成任务相关初始抓握：既考察正负轮轴力矩，也考虑径向/轴向扰动和手部执行限制。可用于构建初态库、筛选接管候选位置，而非要求RL从任意接触开始盲目探索。

## 限制

这是静态抓姿合成，不是闭环抓持控制或整段接触切换策略；不建模物体与环境接触。力边界假设也需要适配六驱动mimic手，几何GWS不能直接当作真实执行器可达到的承载集合。
