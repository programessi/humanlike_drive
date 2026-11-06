"""Local contact readers, mimic initialization and MuJoCo rendering."""
import gymnasium as gym
import mujoco
import numpy as np

class SimulationSupport(gym.Env):
    @property
    def wheel_angle(self):
        return float(self.data.qpos[self.wheel_q])


    def _contact_metrics(self):
        force = np.zeros(2)
        peak = 0.0
        wheel_other_force = 0.0
        vector = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 not in self.wheel_geoms and c.geom2 not in self.wheel_geoms:
                continue
            other = c.geom2 if c.geom1 in self.wheel_geoms else c.geom1
            body = self.model.geom_bodyid[other]
            mujoco.mj_contactForce(self.model, self.data, i, vector)
            norm = float(np.linalg.norm(vector[:3]))
            peak = max(peak, norm)
            hand_contact = False
            for side, bodies in enumerate(self.hand_body_groups):
                if body in bodies:
                    force[side] += norm
                    hand_contact = True
            if not hand_contact:
                wheel_other_force += norm
        self.contact_forces = force
        return force, peak, wheel_other_force


    def _set_initial_mimic(self):
        # This is reset initialization only. Runtime coupling uses equality constraints.
        for side in ("L", "R"):
            for finger in ("index", "middle", "ring", "pinky"):
                self.data.qpos[self.model.joint(f"{side}_{finger}_intermediate_joint").qposadr] = self.data.qpos[self.model.joint(f"{side}_{finger}_proximal_joint").qposadr]
            pitch = self.data.qpos[self.model.joint(f"{side}_thumb_proximal_pitch_joint").qposadr][0]
            for segment, ratio in (("intermediate", 1.6), ("distal", 2.4)):
                self.data.qpos[self.model.joint(f"{side}_thumb_{segment}_joint").qposadr] = ratio*pitch


    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                import threading
                existing_threads=set(threading.enumerate())
                self.viewer = viewer.launch_passive(self.model, self.data)
                self.viewer_thread=next((thread for thread in threading.enumerate()
                                         if thread not in existing_threads and "_launch_internal" in thread.name),None)
                self.viewer.opt.geomgroup[3] = 0
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self.model.camera(self.camera).id
            self.viewer.sync()
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.render_options)
        return self.renderer.render().copy()


    def update_display(self, rgb):
        """Show the actual CARLA RGB on the front display when rendering."""
        from PIL import Image
        if self.render_mode == "human" and self.viewer is None:
            self.render()
        elif self.render_mode != "human" and self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        tex = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "front_rgb")
        width, height = int(self.model.tex_width[tex]), int(self.model.tex_height[tex])
        pixels = np.asarray(Image.fromarray(rgb).resize((width,height)),dtype=np.uint8)
        address = self.model.tex_adr[tex]
        self.model.tex_data[address:address+width*height*3] = pixels.ravel()
        if self.render_mode == "human":
            self.viewer.update_texture(tex)
        else:
            mujoco.mjr_uploadTexture(self.model,self.renderer._mjr_context,tex)


    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            # MuJoCo 3.3's passive viewer closes asynchronously. Let its native
            # GL thread finish before Python tears down model/driver objects.
            if self.viewer_thread is not None:
                self.viewer_thread.join(timeout=5)
                self.viewer_thread = None
            self.viewer = None

