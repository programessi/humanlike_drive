# 原CARLA驾驶基线

本目录是独立Python项目，包名`drive_baseline`。复制本目录并提供Python依赖和机器人源模型后即可运行，不需要manipulation目录。

前视RGB → 视觉路径点 → 路径控制器 → 目标轮角 → 角度反馈与圆弧参考 → IK查表与26维关节残差PPO → 显式限幅PD → 双手接触 → 实际轮角 → CARLA。

## 目录

`src/drive_baseline/`是源码；`assets/`场景，`configs/`配置，`scripts/`工具，`tests/`测试，`runs/`权重，`outputs/`原始评价和视频，`results/`轻量报告，`docs/`历史结果与计划。`data/`、`weights/`、`third_party/`保存视觉数据、ImageNet权重和CARLA依赖。

## 已记录的驾驶结果

`outputs/carla_drive_clockwise/drive.mp4`为300秒历史视频：412.68m、0碰撞、4次压线。前200秒轮角RMSE0.60°，后100秒14.78°。该结果未通过长时稳定驾驶验收。轻量记录见`results/carla_drive_clockwise/`。

历史驾驶使用的视觉模型哈希为`d462b6629da0cc69cd6e16b9f75fd0131548a1115d7975d4e39499e73f27348e`；目前`runs/vision_routes/best.pt`已更新至`f7bb8f16b5a3dccd39ae0c5dffb0fbcdbec425dd5e5b66bf95d10f1c5ee76b27`。旧视觉checkpoint未单独保留，不能声称当前模型逐帧复现该视频。

## 运行

从本目录执行。`run.sh`配置本项目源码路径并使用现有mujoco环境；不重新安装环境。

```bash
bash scripts/run.sh -m drive_baseline.evaluate \
  --mode sine --amplitude 10 --seconds 10 --output outputs/recheck_new

# 连接或启动CARLA，使用当前视觉权重运行同类基线结构
bash scripts/demo.sh \
  --vision-policy runs/vision_routes/best.pt --navigation route \
  --steps 3000 --wheel-fullscale-deg 90 --adaptive-speed --robot-guard \
  --target-rate-deg 20 --integral-gain 1 --stop-on-collision \
  --video --output outputs/baseline_recheck_new

bash scripts/run.sh -m unittest discover -s tests -v
```

CARLA服务端在`third_party/vendor/carla-0.9.15`，API在`third_party/vendor/carla-python`；运行桥接时用`bash scripts/with_carla.sh -m drive_baseline.carla_bridge ...`。默认演示脚本仍是早期60秒配置，上述命令显式选择长程参数。

依赖见`pyproject.toml`，默认机器人源模型位于`/home/nubot/unitree/unitree_ros/robots/h1_2_description/`。未实现实物部署、自主抓取或换手。

详细历史结果：[RESULTS_HISTORY.md](docs/RESULTS_HISTORY.md)；独立操作测试：[报告](results/isolated_tests_20261004/README.md)。历史记录中的旧源码路径属于当时版本，新的运行路径以本页为准。
