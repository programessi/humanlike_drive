"""Synchronous CARLA collection and driving through the physical steering wheel."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import queue
import time

import imageio.v2 as imageio
import numpy as np

from .env import SteeringEnv, TaskConfig
from .vision import LOOKAHEAD_METRES


def speed_mps(vehicle):
    v=vehicle.get_velocity()
    return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)


def camera_frame(images,frame,timeout=5):
    deadline=time.monotonic()+timeout
    while True:
        image=images.get(timeout=max(.01,deadline-time.monotonic()))
        if image.frame==frame:
            array=np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)
            return array[:,:,:3][:,:,::-1].copy()
        if image.frame>frame:
            raise RuntimeError(f"Camera frame {image.frame} skipped requested frame {frame}")


def teacher_waypoints(road_map,vehicle):
    transform=vehicle.get_transform()
    wp=road_map.get_waypoint(transform.location)
    yaw=math.radians(transform.rotation.yaw)
    cs,sn=math.cos(yaw),math.sin(yaw)
    xy=[]
    for distance in LOOKAHEAD_METRES:
        candidates=wp.next(distance)
        if not candidates:
            raise RuntimeError("Route reached a road end")
        # Road-following teacher chooses the least heading change at junctions.
        candidate=min(candidates,key=lambda w:abs((w.transform.rotation.yaw-transform.rotation.yaw+180)%360-180))
        dx=candidate.transform.location.x-transform.location.x
        dy=candidate.transform.location.y-transform.location.y
        xy.append([cs*dx+sn*dy,-sn*dx+cs*dy])
    return np.array(xy)


def path_to_steer(xy,max_front_angle,lookahead_index=2,wheelbase=2.8):
    x,y=xy[lookahead_index]
    curvature=2*y/max(x*x+y*y,1e-3)
    return float(np.clip(math.atan(wheelbase*curvature)/max_front_angle,-1,1))


def arc_waypoints_to_xy(lateral,distances=LOOKAHEAD_METRES):
    """Recover forward displacement using a local circular-arc approximation.

    Teacher waypoint.next(s) uses road arc length, not vehicle forward x.
    Solve |y|/s = (1-cos(theta))/theta on its monotonic branch. This is
    exact for a centred circular lane up to theta=2.331 rad; it remains an
    approximation for offsets, junctions and changing curvature.
    """
    distances=np.asarray(distances,dtype=float)
    lateral=np.asarray(lateral,dtype=float)
    lo=np.zeros_like(distances);hi=np.full_like(distances,2.331)
    ratio=np.abs(lateral)/distances
    for _ in range(32):
        theta=(lo+hi)/2
        prediction=(1-np.cos(theta))/theta
        low=prediction<ratio
        lo=np.where(low,theta,lo);hi=np.where(low,hi,theta)
    theta=(lo+hi)/2
    forward=distances*np.sinc(theta/np.pi)
    return np.column_stack([forward,lateral])


def lateral_error(vehicle_transform,lane):
    """Signed horizontal offset from the nearest lane centre, in metres."""
    yaw=math.radians(lane.transform.rotation.yaw)
    dx=vehicle_transform.location.x-lane.transform.location.x
    dy=vehicle_transform.location.y-lane.transform.location.y
    return -math.sin(yaw)*dx+math.cos(yaw)*dy


def safe_speed_target(cruise_speed,steer_target,wheel_error):
    """Slow for a predicted bend and for lag in the physical wheel actuator."""
    turn_speed=cruise_speed/(1+12*abs(steer_target))
    lag_scale=1+5*max(0,abs(wheel_error)-.08)
    return float(min(cruise_speed,max(.6,turn_speed/lag_scale)))


def residual_scale(target,actual):
    """Fade to the IK/PD baseline outside the nominal PPO training envelope."""
    error_scale=np.clip((math.radians(15)-abs(target-actual))/math.radians(8),0,1)
    angle_scale=np.clip((math.radians(45)-max(abs(target),abs(actual)))/math.radians(10),0,1)
    return float(min(error_scale,angle_scale))


def load_visual(path,device):
    import torch
    from .vision import policy_from_checkpoint
    payload=torch.load(path,map_location=device,weights_only=True)
    policy=policy_from_checkpoint(payload).to(device);policy.eval()
    return policy


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=2000)
    p.add_argument("--map",default="Town04")
    p.add_argument("--mode",choices=["collect","teacher-robot","vision-robot","vision-direct"],default="teacher-robot")
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--spawn",type=int,default=0)
    p.add_argument("--speed",type=float,default=3.0)
    p.add_argument("--adaptive-speed",action="store_true",help="Reduce speed using predicted curvature and measured wheel tracking lag")
    p.add_argument("--wheel-fullscale-deg",type=float,default=450.)
    p.add_argument("--wheel-sign",type=float,choices=[-1.,1.],default=1.)
    p.add_argument("--robot-policy",type=Path)
    p.add_argument("--robot-guard",action="store_true",help="Fade PPO residuals for large errors or angles outside the nominal training domain")
    p.add_argument("--target-rate-deg",type=float,default=0,help="Maximum commanded wheel target rate; 0 disables filtering")
    p.add_argument("--integral-gain",type=float,default=0,help="Bounded wheel-error integral in the robot hand reference; old checkpoints use 0")
    p.add_argument("--vision-policy",type=Path)
    p.add_argument("--path-geometry",choices=["fixed","circular"],default="fixed",help="Convert road-distance waypoint labels into vehicle-frame coordinates")
    p.add_argument("--navigation",choices=["none","route"],default="none",help="Optional coarse GPS route instructions; no route coordinates enter the visual policy")
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path,default=Path("outputs/carla"))
    p.add_argument("--video",action="store_true")
    p.add_argument("--stop-on-collision",action="store_true",help="End evaluation on the first collision event")
    p.add_argument("--viewer",action="store_true",help="Show the live MuJoCo cockpit with CARLA RGB; requires a desktop display")
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--collect-noise",type=float,default=.025,help="Teacher collection steering perturbation for recovery examples")
    args=p.parse_args()
    try:
        import carla
    except ImportError:
        raise SystemExit("CARLA Python API missing: use the matching wheel via scripts/with_carla.sh")
    if args.mode.startswith("vision") and not args.vision_policy:
        p.error("vision modes require --vision-policy")
    if args.steps<1 or args.wheel_fullscale_deg<=0 or args.speed<=0 or args.target_rate_deg<0 or args.integral_gain<0:
        p.error("steps, speed and wheel-fullscale-deg must be positive; rate and integral gain must be nonnegative")
    if args.viewer and (args.video or args.mode not in ("teacher-robot","vision-robot")):
        p.error("--viewer requires a robot mode and cannot be combined with --video")
    client=carla.Client(args.host,args.port);client.set_timeout(30)
    world=client.get_world()
    if world.get_map().name.split("/")[-1]!=args.map:
        world=client.load_world(args.map)
    road_map=world.get_map()
    settings=world.get_settings()
    original=world.get_settings()
    settings.synchronous_mode=True;settings.fixed_delta_seconds=.1
    actors=[];env=None;writer=None;manifest=None;log=[];collision_events=[];lane_events=[];completed=False
    rng=np.random.default_rng(args.seed);noise=0.0
    args.output.mkdir(parents=True,exist_ok=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config["server_version"]=client.get_server_version()
    config["policy_sha256"]={name:hashlib.sha256(path.read_bytes()).hexdigest()
                              for name,path in (("robot",args.robot_policy),("vision",args.vision_policy)) if path}
    (args.output/"run_config.json").write_text(json.dumps(config,indent=2))
    started=time.monotonic();distance=0.0
    try:
        world.apply_settings(settings)
        bp=world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name","h1_driver")
        transforms=road_map.get_spawn_points()
        vehicle=world.spawn_actor(bp,transforms[args.spawn%len(transforms)]);actors.append(vehicle)
        bp=world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x","640");bp.set_attribute("image_size_y","360");bp.set_attribute("fov","90")
        camera=world.spawn_actor(bp,carla.Transform(carla.Location(x=.8,z=1.5)),attach_to=vehicle);actors.append(camera)
        images=queue.Queue();camera.listen(images.put)
        collision=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=vehicle)
        actors.append(collision)
        collision.listen(lambda event:collision_events.append({"frame":event.frame,
                          "actor_type":getattr(event.other_actor,"type_id","unknown"),
                          "normal_impulse_Ns":[event.normal_impulse.x,event.normal_impulse.y,event.normal_impulse.z]}))
        invasion=world.spawn_actor(world.get_blueprint_library().find("sensor.other.lane_invasion"),carla.Transform(),attach_to=vehicle)
        actors.append(invasion)
        invasion.listen(lambda event:lane_events.append({"frame":event.frame,
                        "markings":[str(marking.type).split(".")[-1] for marking in event.crossed_lane_markings]}))
        physics=vehicle.get_physics_control()
        max_front_angle=math.radians(max(w.max_steer_angle for w in physics.wheels))
        if max_front_angle<=0:raise RuntimeError("Vehicle has no steering wheels")
        config["max_front_angle_degrees"]=math.degrees(max_front_angle)
        config["camera"]={"width":640,"height":360,"fov_degrees":90,"mount_x_m":.8,"mount_z_m":1.5}
        (args.output/"run_config.json").write_text(json.dumps(config,indent=2))
        initial_frame=world.tick()
        rgb=camera_frame(images,initial_frame)
        image_frame=initial_frame
        previous_location=vehicle.get_location()
        robot_policy=None;visual=None;obs=None;navigator=None
        previous_wheel_target=0.0
        if args.mode in ("teacher-robot","vision-robot"):
            env=SteeringEnv(TaskConfig(episode_seconds=max(20,args.steps*.1+1),integral_gain=args.integral_gain),
                            render_mode="human" if args.viewer else "rgb_array" if args.video else None,width=960,height=540)
            obs,_=env.reset(seed=7)
            previous_wheel_target=env.wheel_angle
            if args.robot_policy:
                from stable_baselines3 import PPO
                robot_policy=PPO.load(args.robot_policy,device="cpu")
        if args.mode.startswith("vision"):
            visual=load_visual(args.vision_policy,args.device)
        if args.navigation=="route":
            from .navigation import RouteNavigator
            navigator=RouteNavigator.from_map(road_map,vehicle.get_location())
            (args.output/"navigation_route.json").write_text(json.dumps(navigator.points.tolist()))
        if args.mode=="collect":
            (args.output/"images").mkdir(exist_ok=True)
            manifest=(args.output/"manifest.jsonl").open("a")
        if args.video:
            writer=imageio.get_writer(args.output/"drive.mp4",fps=10,macro_block_size=1)
        for step in range(args.steps):
            loop_started=time.monotonic()
            speed=speed_mps(vehicle)
            navigation_command=0
            # RGB input corresponds to the current vehicle state. Only the
            # collection/teacher modes query map data to choose a target.
            if visual is not None:
                import torch
                from .vision import image_tensor
                if navigator is not None:
                    pose=vehicle.get_transform()
                    navigation_command=navigator.command(pose.location.x,pose.location.y,pose.rotation.yaw)
                command=torch.zeros((1,3),device=args.device);command[0,navigation_command]=1
                with torch.no_grad():
                    y=visual(image_tensor(rgb)[None].to(args.device),torch.tensor([speed],device=args.device),command)[0].cpu().numpy()
                xy=y if visual.predict_xy else arc_waypoints_to_xy(y) if args.path_geometry=="circular" else np.column_stack([LOOKAHEAD_METRES,y])
            else:
                if navigator is None:
                    xy=teacher_waypoints(road_map,vehicle)
                else:
                    pose=vehicle.get_transform()
                    xy=navigator.path_relative(pose.location.x,pose.location.y,pose.rotation.yaw,LOOKAHEAD_METRES)
                    from .navigation import turn_command
                    navigation_command=turn_command(*xy[-1])
            steer_target=path_to_steer(xy,max_front_angle)
            fullscale=math.radians(args.wheel_fullscale_deg)
            unclipped_target=args.wheel_sign*steer_target*fullscale
            requested_target=float(np.clip(unclipped_target,-.95,.95))
            wheel_target=requested_target
            target_velocity=0.0;policy_scale=0.0
            if env is not None and args.target_rate_deg:
                delta=float(np.clip(wheel_target-previous_wheel_target,-math.radians(args.target_rate_deg)*.1,math.radians(args.target_rate_deg)*.1))
                wheel_target=previous_wheel_target+delta
                target_velocity=delta/.1
            previous_wheel_target=wheel_target
            if env is not None:
                obs=env.set_target(wheel_target,target_velocity)
                for _ in range(5):
                    action=np.zeros(26) if robot_policy is None else robot_policy.predict(obs,deterministic=True)[0]
                    policy_scale=residual_scale(wheel_target,env.wheel_angle) if args.robot_guard else 1.0
                    action=policy_scale*action
                    obs,_,terminated,_,info=env.step(action)
                    if terminated:raise RuntimeError(f"Robot steering failure: {info}")
                steer=float(np.clip(args.wheel_sign*env.wheel_angle/fullscale,-1,1))
                physical_angle=env.wheel_angle
            else:
                steer=steer_target;physical_angle=None
                if args.mode=="collect":
                    if step%10==0:noise=float(rng.normal(0,args.collect_noise))
                    steer=float(np.clip(steer+noise,-1,1))
            wheel_error=0 if env is None else requested_target-physical_angle
            speed_target=safe_speed_target(args.speed,steer_target,wheel_error) if args.adaptive_speed else args.speed
            acceleration=np.clip(.4*(speed_target-speed),-.8,.5)
            vehicle.apply_control(carla.VehicleControl(steer=steer,throttle=float(max(0,acceleration)),brake=float(max(0,-acceleration))))
            # Label the current RGB/state before advancing CARLA.
            if manifest:
                name=f"images/{image_frame:08d}.png"
                imageio.imwrite(args.output/name,rgb)
                manifest.write(json.dumps({"image":name,"frame":image_frame,"speed":speed,"command":navigation_command,
                                           "lateral_waypoints":xy[:,1].tolist(),"path_waypoints":xy.tolist(),
                                           "map":args.map,"source":"CARLA coherent planned route" if navigator is not None else "CARLA waypoint teacher"})+"\n")
            frame=world.tick();rgb=camera_frame(images,frame)
            image_frame=frame
            # Evaluation may use map ground truth even in vision mode, but it
            # never feeds these measurements into the visual policy.
            transform=vehicle.get_transform()
            server_control=vehicle.get_control()
            distance+=math.hypot(transform.location.x-previous_location.x,transform.location.y-previous_location.y)
            previous_location=transform.location
            lane=road_map.get_waypoint(transform.location)
            lane_error=transform.location.distance(lane.transform.location)
            row={"frame":frame,"time":(step+1)*.1,"speed":speed,"target_speed":speed_target,"target_steer":steer_target,"applied_steer":steer,
                 "server_steer":server_control.steer,
                 "target_wheel_rad":wheel_target,"actual_wheel_rad":physical_angle,"lane_error_m":lane_error,
                 "requested_wheel_rad":requested_target,"unclipped_wheel_rad":unclipped_target,"robot_policy_scale":policy_scale,
                 "navigation_command":navigation_command,"navigation_offset_m":navigator.offset if navigator is not None else None,
                 "lateral_error_m":lateral_error(transform,lane),"distance_m":distance,
                 "road_id":lane.road_id,"lane_id":lane.lane_id,
                 "collisions":len(collision_events),"lane_invasions":len(lane_events)}
            if env is not None:
                row.update(left_hand_force_N=info["left_force"],right_hand_force_N=info["right_force"],
                           robot_peak_force_N=info["peak_force"],robot_error_deg=info["error_deg"])
            log.append(row)
            if args.viewer:
                env.update_display(rgb);env.render()
                if not env.viewer.is_running():break
                time.sleep(max(0,.1-(time.monotonic()-loop_started)))
            if writer:
                if env is not None:
                    from PIL import Image, ImageDraw
                    env.update_display(rgb)
                    canvas=Image.new("RGB",(1600,540),(15,18,22))
                    canvas.paste(Image.fromarray(env.render()),(0,0))
                    canvas.paste(Image.fromarray(rgb),(960,80))
                    draw=ImageDraw.Draw(canvas)
                    source=f"RGB + NAV: {['STRAIGHT','LEFT','RIGHT'][navigation_command]}" if navigator is not None else "RGB road following" if visual is not None else "MAP TEACHER (integration)"
                    draw.text((975,20),f"CARLA | {speed:.1f} m/s | {source}",fill=(240,240,240))
                    draw.text((975,460),f"Wheel target {math.degrees(wheel_target):+.1f} deg / actual {math.degrees(physical_angle):+.1f} deg",fill=(240,240,240))
                    writer.append_data(np.asarray(canvas))
                else:
                    writer.append_data(rgb)
            if step%100==0:print(row,flush=True)
            if args.stop_on_collision and collision_events:break
        completed=len(log)==args.steps
    finally:
        if log:
            with (args.output/"drive_metrics.csv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
            summary={"mode":args.mode,"frames":len(log),"requested_frames":args.steps,
                            "completed":completed,"collisions":len(collision_events),"lane_invasions":len(lane_events),
                            "first_collision_time_s":next((r["time"] for r in log if r["collisions"]),None),
                            "lane_error_mean_m":float(np.mean([r["lane_error_m"] for r in log])),
                            "lateral_error_abs_mean_m":float(np.mean([abs(r["lateral_error_m"]) for r in log])),
                            "lateral_error_abs_max_m":float(np.max([abs(r["lateral_error_m"]) for r in log])),
                            "distance_m":distance,"speed_mean_mps":float(np.mean([r["speed"] for r in log])),
                            "sim_seconds":len(log)*.1,"wall_seconds":time.monotonic()-started,
                            "navigation":args.navigation,
                            "lane_marking_events":dict(Counter(marking for event in lane_events for marking in event["markings"])),
                            "wheel_fullscale_deg":args.wheel_fullscale_deg,
                            "target_clipped_fraction":float(np.mean([abs(r["unclipped_wheel_rad"])>.95 for r in log])),
                            "target_rate_limited_fraction":float(np.mean([abs(r["requested_wheel_rad"]-r["target_wheel_rad"])>1e-6 for r in log]))}
            if env is not None:
                summary.update(wheel_tracking_rmse_deg=float(np.sqrt(np.mean([r["robot_error_deg"]**2 for r in log]))),
                               sampled_hand_contact_fraction=float(np.mean([r["left_hand_force_N"]+r["right_hand_force_N"]>1 for r in log])),
                               sampled_peak_contact_force_N=float(max(r["robot_peak_force_N"] for r in log)),
                               wheel_to_vehicle_max_error=float(max(abs(r["server_steer"]-np.clip(args.wheel_sign*r["actual_wheel_rad"]/math.radians(args.wheel_fullscale_deg),-1,1)) for r in log)))
            if navigator is not None:
                summary.update(route_offset_mean_m=float(np.mean([r["navigation_offset_m"] for r in log])),
                               route_offset_max_m=float(max(r["navigation_offset_m"] for r in log)),
                               route_progress_m=navigator.progress,
                               navigation_command_counts=dict(Counter(r["navigation_command"] for r in log)))
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
            (args.output/"collision_events.json").write_text(json.dumps(collision_events,indent=2))
            (args.output/"lane_events.json").write_text(json.dumps(lane_events,indent=2))
        if manifest:manifest.close()
        if writer:writer.close()
        if env is not None:env.close()
        for actor in reversed(actors):
            if hasattr(actor,"stop"):actor.stop()
            actor.destroy()
        world.apply_settings(original)


if __name__=="__main__":main()
