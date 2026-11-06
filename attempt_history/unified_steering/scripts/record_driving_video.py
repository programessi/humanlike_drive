"""Record frozen-policy evaluations without changing the physical executor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from unified_steering.structured import driving_learning as learning

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASH = 'bc2699474b133f26135ae36b354865a6291b31d944d38361da7ef547aa58451b'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


class Video:
    def __init__(self, env, output, stage):
        self.env, self.output, self.stage = env, output, stage
        self.frames, self.steps = 0, 0
        self.actual, self.times = [], []
        self.fonts = {s: ImageFont.truetype(FONT, s) for s in (15, 18, 21, 26)}
        self.renderer = mujoco.Renderer(env.model, height=480, width=640)
        self.close_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.close_camera)
        self.close_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.close_camera.lookat[:] = env.controller.center
        self.close_camera.azimuth = 35
        self.close_camera.elevation = -30
        self.close_camera.distance = .8
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '20',
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'demo_4x.mp4')],
            stdin=subprocess.PIPE)

    def capture(self, info=None):
        env = self.env
        angle, target = np.rad2deg([env.wheel_angle, env.target_angle])
        self.times.append(env.time)
        self.actual.append(angle)
        if self.steps % 4 and not (info and info.get('termination_reason')):
            return
        image = Image.new('RGB', (1280, 720), (15, 22, 33))
        for x, camera in [(0, self.close_camera), (640, 'hands')]:
            self.renderer.update_scene(env.data, camera=camera, scene_option=env.render_options)
            view = ImageEnhance.Brightness(Image.fromarray(self.renderer.render().copy())).enhance(1.25)
            image.paste(view, (x, 62))
        draw = ImageDraw.Draw(image)
        def text(pos, value, size=18, color=(226, 234, 243)):
            draw.text(pos, value, font=self.fonts[size], fill=color)
        title = ('R1  双手90°双向跟踪、保持与回正' if self.stage == 'r1'
                 else 'R2  180°提前释放、接管与受控回正')
        text((20, 10), title, 26)
        text((910, 17), '冻结策略 v121  |  4×播放', 21, (121, 221, 187))
        for x, label in [(14, '轮盘与手部'), (654, '双臂视角')]:
            draw.rounded_rectangle((x, 74, x + 128, 104), radius=5, fill=(16, 25, 38))
            text((x + 10, 76), label, 18)
        text((20, 552), f'目标 {target:+7.2f}°    实际 {angle:+7.2f}°    误差 {target-angle:+6.2f}°', 21)
        force = env.contact_forces
        text((20, 588), f'接触力   左 {force[0]:5.1f} N    右 {force[1]:5.1f} N', 21)
        load = env.load.centering_torque(env.wheel_angle)
        margin = info['physics_min_arm_margin_deg'] if info else np.rad2deg(env._arm_margin())
        text((20, 623), f'回正力矩 {load:+.3f} N·m    臂余量 {margin:.2f}°', 21)
        text((20, 667), f'仿真时钟 {env.time:6.1f} / {env.config.episode_seconds:.1f} s  ·  峰值参考速度 5°/s', 18)
        text((760, 548), '角度轨迹：灰色目标 / 绿色实际', 18)
        x0, y0, width, height = 801, 585, 452, 97
        amplitude = env.config.amplitude_deg + 8
        def point(t, a):
            return (x0 + t / env.config.episode_seconds * width,
                    y0 + height / 2 - a / amplitude * height / 2)
        for a in [-env.config.amplitude_deg, 0, env.config.amplitude_deg]:
            y = point(0, a)[1]
            draw.line((x0, y, x0 + width, y), fill=(48, 60, 75))
            text((748, y - 10), f'{a:+.0f}°', 15, (153, 168, 185))
        reference_times = np.arange(len(env.command_angles)) * env.dt
        mask = reference_times <= env.config.episode_seconds
        ref = [point(t, a) for t, a in zip(reference_times[mask][::4], np.rad2deg(env.command_angles[mask][::4]))]
        draw.line(ref, fill=(134, 145, 162), width=2)
        path = [point(t, a) for t, a in zip(self.times[::4], self.actual[::4])]
        if len(path) > 1:
            draw.line(path, fill=(66, 218, 170), width=3)
        cx, cy = point(env.time, angle)
        draw.ellipse((cx-3, cy-3, cx+3, cy+3), fill=(255, 190, 85))
        self.process.stdin.write(image.tobytes())
        self.frames += 1
        if self.frames in (1, 200, 500, 900):
            image.save(self.output / f'frame_{self.frames:04d}.jpg', quality=94)
        if self.steps % 400 == 0:
            print(f'VIDEO_PROGRESS {self.stage}: sim={env.time:.1f}s frames={self.frames}', flush=True)

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.renderer.close()
        if code:
            raise RuntimeError(f'ffmpeg failed: {code}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['r1', 'r2'], required=True)
    parser.add_argument('--output-root', type=Path, help='Fresh output directory; existing videos are never overwritten')
    args = parser.parse_args()
    root = args.output_root or ROOT / 'results/DRIVING_R0_R2_ACCEPTANCE_V121/videos'
    output = root / args.stage
    output.mkdir(parents=True, exist_ok=False)
    model = ROOT / 'runs/driving_r2_frozen_v121/teacher.zip'
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != EXPECTED_HASH:
        raise ValueError('Frozen policy hash changed')
    suite = ('driving_r1_exact72_retention_v121' if args.stage == 'r1'
             else 'driving_r2_independent_v121')
    original = ROOT / 'results' / suite
    protocol = json.loads((original / 'protocol.json').read_text())
    cases = json.loads((original / 'summary.json').read_text())['cases']
    chosen = next(c for c in cases if c['amplitude_deg'] == (90 if args.stage == 'r1' else 180)
                  and c['direction'] == -1 and c['template'] == 'sequence'
                  and c['seed'] == (43001 if args.stage == 'r1' else 56051)
                  and (len(c['case']) < 5 or c['case'][4].get('phase_deg', 0) == 0))
    for name in ['cases', 'traces']:
        (output / name).mkdir()
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    base = learning.DrivingEnv
    class RecordingEnv(base):
        def reset(self, *a, **kw):
            result = super().reset(*a, **kw)
            self.video = Video(self, output, args.stage)
            self.video.capture()
            return result
        def step(self, action):
            result = super().step(action)
            self.video.steps += 1
            self.video.capture(result[-1])
            return result
        def close(self):
            if hasattr(self, 'video'):
                self.video.close()
                del self.video
            super().close()
    learning.DrivingEnv = RecordingEnv
    learning.init_worker(model, protocol, output, False, 0.)
    result = learning.run_job(chosen['case'])
    if not result['passed']:
        raise RuntimeError('Recorded rollout failed original acceptance gates')
    record_path = next((output / 'cases').glob('*.json'))
    trace_name = record_path.with_suffix('.npz').name
    differences = {}
    reference_trace = original / 'traces' / trace_name
    if not reference_trace.exists():
        reference_trace = ROOT / 'results/DRIVING_R0_R2_ACCEPTANCE_V121/videos' / args.stage / 'traces' / trace_name
    with np.load(reference_trace) as before, np.load(output / 'traces' / trace_name) as after:
        for key in ['time', 'angle_deg', 'target_deg', 'forces_n', 'torques_nm', 'active_q']:
            if before[key].shape != after[key].shape:
                raise RuntimeError(f'Changed trace shape: {key}')
            differences[key] = float(np.max(np.abs(before[key] - after[key])))
    if any(value != 0 for value in differences.values()):
        raise RuntimeError(f'Recorded rollout differs from acceptance trace: {differences}')
    if hashlib.sha256(model.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen checkpoint changed during recording')
    metadata = dict(model=str(model.relative_to(ROOT)), model_sha256=digest,
        original_suite=str(original.relative_to(ROOT)), comparison_trace_source=str(reference_trace.relative_to(ROOT)), case=chosen['case'],
        playback_speed=4, output_fps=20, capture_every_policy_steps=4,
        source_policy_hz=20, policy_updated=False, script_at_execution=False,
        physical_settings_changed=False, display_only='Two cameras, brightness, telemetry overlay',
        original_trace_max_absolute_difference=differences, result=result,
        scope='90deg both-hand or180deg early-regrasp variant; not360/540deg')
    (output / 'video_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print('VIDEO_COMPLETE ' + json.dumps(dict(stage=args.stage, passed=True,
        motion_rmse_deg=result['motion_rmse_deg'], original_trace_max_absolute_difference=differences,
        video=str(output / 'demo_4x.mp4')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
