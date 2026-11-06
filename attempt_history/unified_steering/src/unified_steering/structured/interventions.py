"""Explicit frozen-policy diagnostics. These overrides never qualify a policy."""
from dataclasses import dataclass, replace

import numpy as np

from .env import TaskConfig
from .parking import ParkedController, SupportTeacherEnv


INTERVENTIONS=("none","radial_zero","axial_zero","hand_initial","radial_zero_hand_initial",
               "hand_lead_0p15","hand_lead_0p233")
TRACE_FIELDS=("audit_radius_m","audit_axial_m","audit_target_radius_m","audit_target_axial_m",
              "audit_contact_torque_nm","audit_hand_q","audit_hand_command","audit_filtered_wrist",
              "audit_filtered_hand","audit_workspace_clips","audit_hand_lead_hits","audit_hand_lead_rad")


@dataclass
class DiagnosticConfig(TaskConfig):
    intervention: str = "none"
    intervention_side: int = 1
    intervention_start_seconds: float = 0.


def intervention_action(action, intervention, side):
    result=np.asarray(action,dtype=float).copy()
    if intervention in ("radial_zero","radial_zero_hand_initial"):
        result[side*6+1]=0.
    if intervention=="axial_zero":
        result[side*6+2]=0.
    return result


class DiagnosticController(ParkedController):
    def reset_reference(self, data):
        super().reset_reference(data)
        self.audit_workspace_clips=np.zeros(2,dtype=int)
        self.audit_hand_lead_hits=np.zeros(2,dtype=int)
        self.diagnostic_fixed_hand=None
        self.diagnostic_hand_lead=None

    def _bounds(self, previous, speed, acceleration, limits, measured, dt):
        lead=getattr(self,"diagnostic_hand_lead",None)
        if previous.start!=14 or lead is None:
            return super()._bounds(previous,speed,acceleration,limits,measured,dt)
        original=self.config
        self.config=replace(original,joint_lead=lead)
        try:
            return super()._bounds(previous,speed,acceleration,limits,measured,dt)
        finally:
            self.config=original

    def hand_targets(self, action):
        result=super().hand_targets(action)
        override=getattr(self,"diagnostic_fixed_hand",None)
        if override is not None:
            side,anchor=override
            result[side*6:(side+1)*6]=anchor
        return result

    def update(self, data, action, dt):
        # Observe the same cylindrical projection before the real update;
        # neither targets nor physical state are changed by instrumentation.
        c=self.config
        filtered=self.filtered_action+(1-np.exp(-dt/c.action_filter_seconds))*(np.clip(action,-1,1)-self.filtered_action)
        for side in range(2):
            local=self.axis_frame.T@(self.wrist_positions[side]-self.center)
            rho=max(np.linalg.norm(local[1:]),c.radial_min_m)
            raw_rho=rho+dt*filtered[6*side+1]*c.wrist_linear_speed
            raw_axial=local[0]+dt*filtered[6*side+2]*c.wrist_linear_speed
            self.audit_workspace_clips[side]+=int(raw_rho!=np.clip(raw_rho,c.radial_min_m,c.radial_max_m)
                                                  or raw_axial!=np.clip(raw_axial,c.axial_min_m,c.axial_max_m))
        super().update(data,action,dt)
        difference=abs(self.target[14:]-data.qpos[self.hand_q]).reshape(2,6)
        lead=getattr(self,"hand_joint_lead_rad",None) or self.diagnostic_hand_lead or c.joint_lead
        self.audit_hand_lead_hits+=np.sum(difference>=lead-1e-5,axis=1)


class DiagnosticEnv(SupportTeacherEnv):
    controller_class=DiagnosticController

    def __init__(self, config=None, **kwargs):
        config=config or DiagnosticConfig()
        if config.intervention not in INTERVENTIONS or config.intervention_side not in (0,1):
            raise ValueError("Unknown diagnostic intervention or side")
        super().__init__(config,**kwargs)

    def reset(self, *args, **kwargs):
        observation,info=super().reset(*args,**kwargs)
        self.diagnostic_hand_anchor=self.controller.target[14:].reshape(2,6).copy()
        return observation,info

    def step(self, action):
        c=self.controller
        active=self.time>=self.config.intervention_start_seconds-1e-9
        intervention=self.config.intervention if active else "none"
        side=self.config.intervention_side
        c.diagnostic_fixed_hand=(side,self.diagnostic_hand_anchor[side]) if intervention in ("hand_initial","radial_zero_hand_initial") else None
        c.diagnostic_hand_lead={"hand_lead_0p15":.15,"hand_lead_0p233":.233}.get(intervention)
        observation,reward,terminated,truncated,info=super().step(intervention_action(action,intervention,side))
        local=(self.data.site_xpos[c.grip_ids]-c.center)@c.axis_frame
        target_local=(c.wrist_positions-c.center)@c.axis_frame
        info.update(audit_radius_m=np.linalg.norm(local[:,1:],axis=1).tolist(),
                    audit_axial_m=local[:,0].tolist(),
                    audit_target_radius_m=np.linalg.norm(target_local[:,1:],axis=1).tolist(),
                    audit_target_axial_m=target_local[:,0].tolist(),
                    audit_contact_torque_nm=self._wheel_contact_torques().tolist(),
                    audit_hand_q=self.data.qpos[c.hand_q].reshape(2,6).tolist(),
                    audit_hand_command=c.target[14:].reshape(2,6).tolist(),
                    audit_filtered_wrist=c.filtered_action[:12].reshape(2,6).tolist(),
                    audit_filtered_hand=c.filtered_action[12:].reshape(2,6).tolist(),
                    audit_workspace_clips=c.audit_workspace_clips.tolist(),
                    audit_hand_lead_hits=c.audit_hand_lead_hits.tolist(),
                    audit_hand_lead_rad=getattr(c,"hand_joint_lead_rad",None) or c.diagnostic_hand_lead or c.config.joint_lead)
        return observation,reward,terminated,truncated,info
