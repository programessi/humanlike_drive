"""Export cockpit snapshots or open the MuJoCo viewer."""
import argparse
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

from .env import SteeringEnv
from .scene import build_scene


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("outputs/cockpit"))
    p.add_argument("--viewer",action="store_true")
    p.add_argument("--rebuild",action="store_true")
    args=p.parse_args()
    if args.rebuild:build_scene()
    env=SteeringEnv(render_mode="human" if args.viewer else "rgb_array",width=1280,height=720)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        env.reset(seed=7)
        if args.viewer:
            env.set_target(0)
            env.render()
            while env.viewer.is_running():
                start=time.monotonic()
                env.step(np.zeros(26));env.render()
                time.sleep(max(0,env.dt-(time.monotonic()-start)))
        else:
            for camera in ("overview","hands","driver"):
                env.camera=camera
                path=args.output/f"{camera}.png"
                imageio.imwrite(path,env.render());print(path)
    finally:env.close()


if __name__=="__main__":main()
