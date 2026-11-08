"""Interactive scene inspection, static renders, and physical settling audit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from humanlike_drive.env.logitech_scene import SCENE, ARM_NAMES
from humanlike_drive.controller.basic_grasps import PRESETS, PRESET_FILE, apply
from humanlike_drive.controller.grasp_contacts import contact_distribution

CAMERAS = ("cabin", "overview", "driver_eye", "hands", "driver", "wheel", "grip_side", "grip_rear")


def add_contact_markers(scene, model, data):
    """Mark loaded contacts; magenta explicitly warns of non-rim loading."""
    for side in contact_distribution(model, data).values():
        for point in side["points"]:
            if scene.ngeom >= scene.maxgeom:
                return
            name = point["link"]
            color = ([1., .75, .05, 1.] if "thumb" in name else
                     [.05, .8, 1., 1.] if "wrist_yaw" in name else
                     [.2, 1., .2, 1.] if "_proximal" in name else [1., .2, .25, 1.])
            if point["wheel_part"] != "rim":
                color = [1., .05, 1., 1.]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([.0028] * 3), np.array(point["position_world_m"]), np.eye(3).ravel(),
                               np.array(color, dtype=np.float32))
            scene.ngeom += 1


def render(output):
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=900, width=1200)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    for name in PRESETS:
        apply(model, data, name)
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam, scene_option=option)
            Image.fromarray(renderer.render()).save(output / f"{name}_{cam}.png")
        renderer.update_scene(data, camera="driver", scene_option=option)
        alpha = model.geom_rgba[:, 3].copy()
        model.geom_rgba[model.geom_group == 1, 3] = .28
        renderer.update_scene(data, camera="driver", scene_option=option)
        add_contact_markers(renderer.scene, model, data)
        Image.fromarray(renderer.render()).save(output / f"{name}_contacts.png")
        renderer.update_scene(data, camera="grip_rear", scene_option=option)
        add_contact_markers(renderer.scene, model, data)
        Image.fromarray(renderer.render()).save(output / f"{name}_grip_rear_contacts.png")
        model.geom_rgba[:, 3] = alpha
    # Remove robot only for CAD/collision alignment pictures.
    apply(model, data, "nine_three")
    model.geom_rgba[:][np.array([model.geom(i).name.startswith(("L_", "R_", "left_", "right_", "pelvis", "torso"))
                                for i in range(model.ngeom)]), 3] = 0
    renderer.update_scene(data, camera="wheel", scene_option=option)
    Image.fromarray(renderer.render()).save(output / "g29_visual.png")
    option.geomgroup[3] = 1
    # Switch off coincident CAD surfaces to avoid z-fighting on close-fitting hulls.
    option.geomgroup[1] = 0
    renderer.update_scene(data, camera="wheel", scene_option=option)
    Image.fromarray(renderer.render()).save(output / "g29_collision_overlay.png")
    renderer.close()
    contact = Image.new("RGB", (1200, 980), "#15212d")
    draw = ImageDraw.Draw(contact)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    for i, name in enumerate(PRESETS):
        tile = Image.open(output / f"{name}_hands.png").resize((600, 450))
        x, y = (i % 2) * 600, (i // 2) * 490
        contact.paste(tile, (x, y + 35))
        draw.text((x + 15, y + 7), f"{i + 1}. {name}", fill="white", font=font)
    contact.save(output / "grasps_overview.png")
    images = [("nine_three_cabin", "坐姿驾驶室"), ("nine_three_overview", "整体布局"),
              ("nine_three_driver_eye", "驾驶员视野"), ("g29_visual", "原始 G29 外观"),
              ("g29_collision_overlay", "碰撞体视图：绿色轮圈，粉色后方拨片，橙色轮辐与轮毂（与上图对照）"),
              ("grasps_overview", "四种初始接触"), ("nine_three_driver", "9 点／3 点抓握近景")]
    images.extend([("nine_three_grip_side", "1号左手侧面：检查轮圈包络"),
                   ("nine_three_grip_rear", "1号左手背面：检查无名指、小指与后方拨片间隙"),
                   ("nine_three_contacts", "半透明轮圈受力点：掌部蓝色、拇指黄色、近端绿色、末端红色；非轮圈受力为品红色"),
                   ("nine_three_grip_rear_contacts", "背面实际受力点：确认四指支撑轮圈；非轮圈受力为品红色")])
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>H1-2 / G29 驾驶室</title><style>body{background:#17212b;color:#eee;font:18px sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;line-height:1.6}img{width:100%;border-radius:8px}h2{margin-top:40px}code{color:#9bd9df}</style><h1>H1-2 / Logitech G29 坐姿驾驶室</h1><p>抓握第5版参考真实驾驶员照片与驾驶教学视频抽帧，联合调整1号的手掌朝向、肩肘姿态与六驱动目标，使前臂和手掌顺接；腕部内扣约5°，四指、掌部和拇指支撑轮圈，避开后方拨片。使用原厂机器人与 G29 CAD，底座水平，转轴保留原模型自然倾角；大屏和仪表为静态预览画面。四种抓握经有限电机物理闭合形成，尚未训练转向策略，也未验证外部负载下的承载能力。</p><p>交互查看：<code>bash scripts/run.sh -m humanlike_drive.env_runner.inspect_scene</code><br>1–4 切换抓握；V 切换相机；F 自由视角；C 切换碰撞视图；T 半透明受力点；空格运行／暂停物理；R 重置当前抓握。自由视角下可用鼠标拖动、缩放。</p>'
    html += "".join(f'<h2>{label}</h2><img loading="lazy" src="{name}.png" alt="{label}">' for name, label in images)
    html += '</html>'
    (output / "index.html").write_text(html)
    print(output / "index.html")


def view(preset="nine_three", camera="cabin", scene_file=SCENE, pose_file=PRESET_FILE,
         replay_file=None, replay_speed=1., max_frames=None):
    # Render directly: the installed 3.3.2 native viewer fails to construct its
    # mjvSceneState buffer for this model. Physics and rendering still use MuJoCo.
    import glfw
    model = mujoco.MjModel.from_xml_path(str(scene_file))
    data = mujoco.MjData(model)
    poses = json.loads(Path(pose_file).read_text())
    if "qpos" in poses:
        poses = {"p0_clear": poses}
    if preset not in poses:
        raise ValueError(f"Unknown preset {preset}; available: {list(poses)}")
    apply(model, data, preset, poses)
    state = {"running": False, "preset": preset, "pending": None, "collision": False, "contacts": False,
             "camera": CAMERAS.index(camera), "camera_pending": True,
             "replay_time": 0., "speed": replay_speed}
    replay = None
    if replay_file is not None:
        with np.load(replay_file) as archive:
            replay = {k: archive[k] for k in ("time", "qpos", "qvel", "ctrl", "qfrc_applied")}
    names = list(poses)
    if not glfw.init():
        raise RuntimeError("GLFW could not connect to the display; use --render with MUJOCO_GL=egl for offline previews")
    window = glfw.create_window(1440, 960, "Humanlike Drive | H1-2 / Logitech G29", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Could not create the scene window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    cam = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    original_alpha = model.geom_rgba[:, 3].copy()

    def key(code):
        if ord("1") <= code < ord("1") + min(len(names), 9):
            state["pending"] = names[code - ord("1")]
        elif code == 32:
            state["running"] = not state["running"]
        elif code in (ord("C"), ord("c")):
            state["collision"] = not state["collision"]
        elif code == ord("T"):
            state["contacts"] = not state["contacts"]
        elif code in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state["speed"] = min(16., state["speed"] * 2)
        elif code in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state["speed"] = max(.125, state["speed"] / 2)
        elif code in (ord("R"), ord("r")):
            state["pending"] = state["preset"]
        elif code in (ord("V"), ord("v")):
            state["camera"] = (state["camera"] + 1) % len(CAMERAS)
            state["camera_pending"] = True
        elif code in (ord("F"), ord("f")):
            state["camera_pending"] = "free"
        elif code == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def on_key(win, code, scancode, action, mods):
        if action == glfw.PRESS:
            key(code)

    cursor = list(glfw.get_cursor_pos(window))
    def cursor_position(win, x, y):
        dx, dy = x - cursor[0], y - cursor[1]
        cursor[:] = [x, y]
        left = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if not (left or right or middle):
            return
        if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
            # Start a free view at the current fixed camera, without a jump.
            eye = data.cam_xpos[cam.fixedcamid].copy()
            forward = -data.cam_xmat[cam.fixedcamid].reshape(3, 3)[:, 2]
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = max(.35, float(np.linalg.norm(eye - np.array([.55, 0, 1.]))))
            cam.lookat[:] = eye + cam.distance * forward
            cam.azimuth = np.rad2deg(np.arctan2(-forward[1], -forward[0]))
            cam.elevation = np.rad2deg(np.arcsin(np.clip(forward[2], -1, 1)))
        shift = (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                 or glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        height = max(1, glfw.get_window_size(win)[1])
        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)

    def scroll(win, x, y):
        if cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05 * y, scene, cam)

    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, cursor_position)
    glfw.set_scroll_callback(window, scroll)
    print("1: 9/3 | 2: 10/2 | 3: left support | 4: right support | V: cameras | F: free camera | C: collision view | T: loaded contacts | Space: physics | R: reset | Esc: close", flush=True)
    armv = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES])
    wheelq = model.jnt_qposadr[model.joint("wheel_joint").id]
    try:
        previous = time.monotonic()
        accumulator = 0.
        frames = 0
        while not glfw.window_should_close(window):
            glfw.poll_events()
            start = time.monotonic()
            elapsed = min(.05, start - previous)
            previous = start
            if state["pending"] is not None:
                state["preset"] = state["pending"]
                apply(model, data, state["preset"], poses)
                state["pending"] = None
                state["running"] = False
                state["replay_time"] = 0.
                accumulator = 0.
                print(state["preset"], flush=True)
            if state["camera_pending"]:
                if state["camera_pending"] == "free":
                    mujoco.mjv_defaultCamera(cam)
                    cam.lookat[:] = [.55, 0, 1.]
                    cam.distance = 3.1
                    cam.azimuth = -135
                    cam.elevation = -20
                else:
                    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    cam.fixedcamid = model.camera(CAMERAS[state["camera"]]).id
                state["camera_pending"] = False
                print("Camera:", "free" if cam.type == mujoco.mjtCamera.mjCAMERA_FREE else CAMERAS[state["camera"]], flush=True)
            if replay is not None:
                if state["running"]:
                    state["replay_time"] = min(replay["time"][-1], state["replay_time"] + elapsed * state["speed"])
                    if state["replay_time"] >= replay["time"][-1]:
                        state["running"] = False
                index = min(len(replay["time"])-1, np.searchsorted(replay["time"], state["replay_time"]))
                data.qpos[:] = replay["qpos"][index]
                data.qvel[:] = replay["qvel"][index]
                data.ctrl[:] = replay["ctrl"][index]
                data.qfrc_applied[:] = replay["qfrc_applied"][index]
                mujoco.mj_forward(model, data)
            elif state["running"]:
                accumulator += elapsed
                while accumulator >= model.opt.timestep:
                    # Finite motor commands; only compensate robot arm gravity.
                    data.qfrc_applied[:] = 0
                    data.qfrc_applied[armv] = data.qfrc_bias[armv]
                    mujoco.mj_step(model, data)
                    accumulator -= model.opt.timestep
            option.geomgroup[3] = int(state["collision"])
            option.geomgroup[1] = int(not state["collision"])
            model.geom_rgba[:, 3] = original_alpha
            if state["contacts"]:
                model.geom_rgba[np.isin(model.geom_group, [1, 3]), 3] = .28
            width, height = glfw.get_framebuffer_size(window)
            if width and height:
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(model, data, option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                if state["contacts"]:
                    add_contact_markers(scene, model, data)
                mujoco.mjr_render(viewport, scene, context)
                view_name = "free" if cam.type == mujoco.mjtCamera.mjCAMERA_FREE else CAMERAS[state["camera"]]
                status = "RUNNING" if state["running"] else "PAUSED"
                text = f"H1-2 / Logitech G29 | {state['preset']} | {status}\nView: {view_name} | Wheel: {np.rad2deg(data.qpos[wheelq]):+.2f} deg"
                if replay is not None:
                    text += f"\nRecorded-state replay | t={state['replay_time']:.2f}s | {state['speed']:g}x"
                controls = "1-4: grasp   V: camera   F: free view   C: collision   T: loaded contacts\nSpace: run/pause   R: reset   Esc: close"
                if replay is not None:
                    controls += "\n+/-: playback speed; visualization only, not a new physics trial"
                if state["contacts"]:
                    controls += "\nRim contacts >0.05 N: cyan=palm  gold=thumb  green=proximal  red=distal; magenta=NON-RIM"
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, text, "", context)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, controls, "", context)
                glfw.swap_buffers(window)
                if frames == 0:
                    print("Scene window ready; first frame rendered", flush=True)
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
            time.sleep(max(0, 1 / 60 - (time.monotonic() - start)))
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--preset", default="nine_three")
    parser.add_argument("--poses", type=Path, default=PRESET_FILE)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--camera", choices=CAMERAS, default="cabin")
    args = parser.parse_args()
    if args.render:
        if args.scene != SCENE or args.poses != PRESET_FILE:
            parser.error("Custom --scene/--poses are for the interactive viewer; use P0 trajectory rendering for videos")
        render(args.render)
    else:
        view(args.preset, args.camera, args.scene, args.poses)
