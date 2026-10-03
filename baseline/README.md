# 基线

本版本记录已经能够驾驶的 IK/PD + PPO 残差方法，不将实验性速度参考或拟议的纯 RL 方法混入基线。

控制结构：前视 RGB → 视觉道路路径点 → 几何路径控制器 → 目标方向盘角度 → 角度反馈及有界积分 → 双手圆弧参考 → IK 查表 + PPO 关节修正 → 限幅 PD/重力补偿 → 物理接触 → 实测被动轮盘角度 → CARLA steer。

## 已记录的驾驶实验

`carla_drive_clockwise/` 保存本机 `outputs/carla_drive_clockwise/` 中的轻量指标、事件及运行配置。原始视频在本机 `outputs/carla_drive_clockwise/drive.mp4`，3000 帧、10 fps、300 秒。

- 行驶 412.68 m，0 碰撞、4 次压线（3 Solid、1 Broken）。
- 前 200 s：轮盘 RMSE 0.60°，平均路线偏移 0.31 m。
- 后 100 s：轮盘 RMSE 14.78°，平均路线偏移 3.74 m。
- 机器人 50 Hz，MuJoCo 500 Hz，CARLA 10 Hz。
- 方向盘没有电机；正常运行不直接写入轮盘角度。
- `wheel_fullscale_deg=90` 表示单侧软件映射，不能视为实现了全范围抓握或 Logitech 实机标定。

该实验加载的视觉模型 SHA-256 是 `d462b6629da0cc69cd6e16b9f75fd0131548a1115d7975d4e39499e73f27348e`。训练随后更新了 `runs/vision_routes/best.pt`，最终权重 SHA-256 为 `f7bb8f16b5a3dccd39ae0c5dffb0fbcdbec425dd5e5b66bf95d10f1c5ee76b27`；旧视觉权重未单独保留，因此不能声称当前文件可逐帧复现历史实验。

## 本机运行同类配置

以下命令使用本机现有最新权重，运行相同的基线控制结构；它不是历史驾驶实验的精确 checkpoint 复现。

```bash
bash scripts/demo.sh \
  --vision-policy runs/vision_routes/best.pt --navigation route \
  --steps 3000 --wheel-fullscale-deg 90 \
  --adaptive-speed --robot-guard --target-rate-deg 20 --integral-gain 1 \
  --stop-on-collision --video --output outputs/baseline_recheck
```

`scripts/demo.sh` 的默认参数仍对应早期 60 秒演示；以上参数显式选择长程配置。权重、图像数据、视频和安装包均不纳入 Git。

## 已知边界

固定坐姿、预抓取开始；尚未实现自主接近抓握、完整换手、踏板操作或实机部署。持续握持会变化；当前 IK 表存在解变化较快及局部几何误差，需要在后续方法中单独检验。详细对照见项目根目录的 `RESULTS.md`。
