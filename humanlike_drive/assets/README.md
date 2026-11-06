# 物理资源

- robots：机器人描述、资源索引、必要模型文件；外部网格依赖记录路径、来源与版本。
- objects：方向盘及其他任务物体的几何、惯量和关节模型。
- scenes：组合机器人与物体的仿真场景。

当前资源：

- `robots/h1_2/`：从本机 Unitree H1-2 描述复制的原始XML、URDF与网格；URDF同时作为Inspire双手mimic定义的来源。
- `objects/logitech_g29/`：原始URDF、G29网格、分块凸碰撞网格和来源校验和 `provenance.json`。
- `scenes/logitech_g29.xml`：可独立加载的MuJoCo组合场景，所有网格路径相对此文件解析。
- `scenes/basic_grasps.json`：本场景重新求解并经物理闭合的四种抓握预设；相同姿态也写入XML的keyframe。
- `scenes/textures/`：程序生成的大屏道路预览与仪表图片。

模型来源、原URDF问题与简化见[场景说明](../docs/SCENE_LOGITECH_G29.md)。不复制旧场景、旧抓握数据或权重。场景运行只依赖此目录内资源；重新生成模型时默认从本机源文件读取，也可使用构建命令的路径参数。
