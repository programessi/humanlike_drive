# 驾驶员姿态参考

本次实际查看了下列照片和视频抽帧，用于定性判断肘部弯曲、前臂与手掌的衔接、9点／3点握持朝向。未从透视图片反推人类三维关节角，也未将图片作为动作捕捉数据。

1. Škoda 官方文章 [Are you sitting comfortably? What about safely?](https://www.skoda-storyboard.com/en/skoda-world/are-you-sitting-comfortably-what-about-safely/)，2021-06-22。文章介绍9点／3点握持、弯曲的肘部，以及控制装置应能自然触及。实际查看的[握持近景](skoda_nine_three_preview.jpg)显示前臂与手掌大体顺接；[侧面坐姿](skoda_seating_preview.jpg)用于检查肩肘与方向盘的距离关系。原图来自 [DSC3941-kopie.jpg](https://cdn.skoda-storyboard.com/2021/06/DSC3941-kopie.jpg) 和 [DSC3912-kopie.jpg](https://cdn.skoda-storyboard.com/2021/06/DSC3912-kopie.jpg)，原始文件及页面保存在本目录。
2. Conquer Driving 的 [The ideal driving position - setting the seat, steering wheel and mirrors](https://www.youtube.com/watch?v=5azLjhw5w28)。实际查看了YouTube原视频的多张预览抽帧表，重点保留[4:10握持帧](conquer_frame_250s.jpg)与[9:40坐姿帧](conquer_frame_580s.jpg)。视频中座椅、方向盘距离与肘部弯曲的演示用于辅助比较，本次没有完整播放视频或核验全部解说。

视频帧来自原视频storyboard第3级：320×180像素／帧，3×3排列，相邻帧间隔10秒。原始抽帧表 `conquer_storyboard_*.jpg`、页面和播放器元数据保存在本目录；提取位置见 `outputs/driverlike_grasp/render_compare.py`。这些是来源图片／视频预览帧，未使用生成图片。

参考中采用的原则是放松的肩肘、手掌和前臂自然衔接。机器人手型、轮圈直径和手臂限位与驾驶员不同，最终姿态仍须通过原始机器人模型和轮圈接触检查。1号腕偏≤10°、前臂与手掌夹角≤15°是工程验收门槛，不是上述来源给出的人体标准角度。
