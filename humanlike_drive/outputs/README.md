# 单次运行产物

后续每次运行使用唯一目录：`outputs/YYYY.MM.DD/HH.MM.SS_<name>/`。

准备随运行保存完整解析配置、代码版本/快照、数据清单、随机种子、训练或评估日志、metrics、checkpoints、逐例轨迹及视频。若使用Hydra，可沿用其配置快照机制。评估、采集和数据处理运行同样保留来源信息。

正式模型留在对应运行的checkpoints中，由results记录路径与哈希，避免维护多个含义不明的best/final副本。README以外的运行产物默认不入Git。本轮没有运行产物。
