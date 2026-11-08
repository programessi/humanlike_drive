"""Replay recorded P0 states using the working GLFW scene viewer."""
import argparse
from pathlib import Path
from humanlike_drive.env_runner.inspect_scene import view, CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--camera', default='driver', choices=CAMERAS)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--frames', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.speed <= 0.:
        parser.error('--speed must be positive')
    print('Recorded-state visualization. Space: play/pause; R: restart; +/-: speed.', flush=True)
    view('p0_clear', args.camera, args.run/'scene.xml', args.run/'grasp_used.json',
         replay_file=args.run/'trajectory.npz', replay_speed=args.speed, max_frames=args.frames)


if __name__ == '__main__':
    main()
