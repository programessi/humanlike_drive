# 冻结策略v121视频展示

[打开播放器页面](index.html)。两段视频均为1280×720、H.264、20fps、4倍速；将播放器速度设为0.25×即可按真实仿真时间观看。

- [R1：90°双向跟踪、保持与回正，37秒](r1/demo_4x.mp4)，完整序列0→−90°→0→+90°→0。
- [R2：180°提前释放、接管与回正，71秒](r2/demo_4x.mp4)，完整序列0→−180°→0→+180°→0。选取独立验收中运动RMSE最差的完整连续案例。

两段均重新运行同一冻结权重，参考峰值速度5°/s。与原验收的时间、轮角、目标、双手接触力、承载转矩及实际关节轨迹逐样本最大差均为0；录制期间没有更新策略或改变物理设置。画面仅增加相机、亮度和遥测叠加。

[文件时长、哈希与来源](video_summary.json)；逐例完整指标见各目录video_metadata.json。录制工具位于scripts/record_driving_video.py。

当前180°方案为提前接管变体，视频为仿真展示；360°/540°及实物迁移尚未验证。

清理后重新录制时使用新的输出目录，例如：`MUJOCO_GL=egl PYTHONPATH=src python scripts/record_driving_video.py --stage r2 --output-root /tmp/steering_video_rerun`。工具使用本目录保留的代表轨迹核对一致性。
