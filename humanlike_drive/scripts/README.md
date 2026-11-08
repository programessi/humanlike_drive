# 工具入口

准备放数据转换、示范采集、设备/物理诊断和批量实验工具。核心实现放源码包，脚本作为薄入口。后续根目录可增加train.py、eval.py，对齐Diffusion Policy的使用习惯；本轮不创建无法运行的空入口。

`summarize_layer1.py`按显式案例—目录映射汇总已完成的45例层1验证，写入`results/physical_feasibility/l1_v1/`，保留各例版本、配置／评价哈希和失败项，并检查两条完整180°的CAD与动作重执行结果。它不会按成功与否自动选择结果，也不重新运行仿真。运行：`python scripts/summarize_layer1.py`。
