"""Versioned input conditioning for the same centralized 24-action teacher.

Statistics are fitted to training data only. Migration changes coordinates while
preserving the parent actor function; it introduces no task servo or action mask.
"""
import numpy as np
import torch
from .legacy.regrasp.policy import TeacherPolicy,TeacherFeatures


class DrivingFeatures(TeacherFeatures):
    def __init__(self,observation_space):
        super().__init__(observation_space)
        for key,size in [('actor',152),('privileged',54)]:
            self.register_buffer(key+'_mean',torch.zeros(size))
            self.register_buffer(key+'_scale',torch.ones(size))

    def forward(self,observations):
        raw=observations['privileged']
        o=(observations['actor']-self.actor_mean)/self.actor_scale
        p=(raw-self.privileged_mean)/self.privileged_scale
        return torch.cat([o,self.encoder(p),raw],-1)


class DrivingTeacherPolicy(TeacherPolicy):
    def make_features_extractor(self):
        return DrivingFeatures(self.observation_space)


def normalize_from_dataset(model,dataset):
    """Change input coordinates exactly once, retaining all24 parent outputs."""
    if isinstance(model.policy,DrivingTeacherPolicy):return
    old=model.policy
    policy=DrivingTeacherPolicy(model.observation_space,model.action_space,lambda _:1e-5,**model.policy_kwargs)
    missing,unexpected=policy.load_state_dict(old.state_dict(),strict=False)
    assert not unexpected and set(missing)=={'features_extractor.'+k for k in ['actor_mean','actor_scale','privileged_mean','privileged_scale']} | {'pi_features_extractor.'+k for k in ['actor_mean','actor_scale','privileged_mean','privileged_scale']} | {'vf_features_extractor.'+k for k in ['actor_mean','actor_scale','privileged_mean','privileged_scale']},(missing,unexpected)
    f=policy.features_extractor
    with torch.no_grad():
        for key in ('actor','privileged'):
            mean=torch.from_numpy(np.mean(dataset[key],axis=0,dtype=np.float64).astype(np.float32))
            scale=torch.from_numpy(np.maximum(np.std(dataset[key],axis=0,dtype=np.float64),.02).astype(np.float32))
            linear=policy.mlp_extractor.actor[0] if key=='actor' else f.encoder[0]
            columns=slice(0,len(mean))
            weight=linear.weight[:,columns].clone()
            linear.bias.add_(weight@mean)
            linear.weight[:,columns].mul_(scale)
            getattr(f,key+'_mean').copy_(mean);getattr(f,key+'_scale').copy_(scale)
    model.policy=policy;model.policy_class=DrivingTeacherPolicy
