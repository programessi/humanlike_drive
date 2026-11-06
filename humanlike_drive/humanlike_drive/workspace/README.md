# 实验工作流

参考Diffusion Policy的workspace职责，准备组织RL训练、BC训练、教师学生/蒸馏等工作流，管理采样或数据加载、优化、随机状态、恢复、日志与周期评价。不同算法可以有不同workspace；最终执行策略仍通过统一policy接口加载。
