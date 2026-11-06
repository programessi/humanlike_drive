# 仿真环境

当前 `logitech_scene.py` 从原始 H1-2、Inspire 双手和 G29 CAD 构建坐姿场景；`cabin.py` 生成座椅、仪表台、大屏及座舱。使用方式见[场景说明](../../docs/SCENE_LOGITECH_G29.md)。

准备放置仿真模型加载、物理步进、reset/step及传感反馈。任务语义引用task，动作执行引用controller，仿真场景引用assets。训练课程初态与正式评估初态分别记录。
