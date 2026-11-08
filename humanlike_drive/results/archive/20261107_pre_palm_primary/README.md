# 掌部主导路线之前的90°案例备份

2026-11-07。用户确认：此前成功的90°案例备份即可，不作为新主线。当前路线见[研究计划](../../../docs/plans/RESEARCH_PLAN.md)，备份时原计划保存在`project_snapshot/`中。

## 已保存的内容

完整副本在本机 `outputs/archive/20261107_pre_palm_primary/`，约156 MB（十进制）：

- `runs/B_left90/`、`runs/B_right90/`：场景、抓握、配置、500 Hz轨迹与接触、评价、源码快照、完整视频和预览。
- `project_snapshot/`：备份时的源码包、脚本、测试、包配置、P0初态，以及原计划、协议和决策。含当时工作区未提交修改；实际运行代码以各案例自己的快照和manifest为准。
- `report_snapshot/`：切换路线前的P0报告、指标、图表和接触诊断。其他旧案例原始运行仍在原outputs目录，未重复复制。
- [SHA256清单](manifest.json)：以上181个文件的路径、大小与哈希；本机备份内也保存同一清单。

| 原案例 | 原运动RMSE | 备份视频 | 备份评价 |
| --- | ---: | --- | --- |
| 左转90°往返 | 0.429° | [视频](../../../outputs/archive/20261107_pre_palm_primary/runs/B_left90/replay_driver.mp4) | [评价](../../../outputs/archive/20261107_pre_palm_primary/runs/B_left90/evaluation.json) |
| 右转90°往返 | 0.586° | [视频](../../../outputs/archive/20261107_pre_palm_primary/runs/B_right90/replay_driver.mp4) | [评价](../../../outputs/archive/20261107_pre_palm_primary/runs/B_right90/evaluation.json) |

原始文件留在原位置，已有链接继续可用。[旧报告](../../p0_complete_20261107/README.md)与[旧计划](../../../docs/plans/RESEARCH_PLAN_v2_3_before_palm_primary.md)供追溯。

## 使用边界

原成功结论只对应旧控制、合成负载和验收要求，不证明掌部主导转矩，不计入新P0，不自动作为新教师或BC数据。同条件对照须按新协议重新运行。

备份未移动或删除原文件。场景和源码快照仍可能引用本机机器人／CAD网格路径，因此不是可直接搬到其他电脑的独立资产包。大体积副本在Git忽略的outputs中，仅存本机；本次没有提交或推送GitHub。
