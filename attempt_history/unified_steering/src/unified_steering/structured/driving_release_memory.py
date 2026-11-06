"""Learned release memory retains all qualified R1 actions while holding.

The hold/release distinction is supervised during training. A neural transition
head predicts it at execution; there are no angle comparisons or expert calls.
"""
import torch
from torch import nn
from .driving_intent_state_policy import IntentStatePolicy
from .driving_coordination_policy import CoordinationPolicy

COUNT=18
class ReleaseTransition(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(188,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,COUNT*COUNT))
        self.register_buffer('log_transition_prior',torch.zeros(COUNT,COUNT))
        self.phase_net=nn.Sequential(nn.Linear(5,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,3*COUNT))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():self.net[-1].bias.copy_((2*torch.eye(COUNT)).reshape(-1))
    def forward(self,x):
        previous=x[:,-COUNT:].argmax(-1);row=torch.arange(len(x),device=x.device)
        output=self.net(x[:,:188]).reshape(-1,COUNT,COUNT)[row,previous]
        phase=self.phase_net(x[:,[120,170,183,186,187]]).reshape(-1,3,COUNT)
        for j,s in enumerate([0,15,16]):output=torch.where((previous==s)[:,None],phase[:,j],output)
        return output+self.log_transition_prior[previous]
class ReleaseIntent(nn.Module):
    def __init__(self):
        super().__init__();self.experts=nn.ModuleList([nn.Sequential(nn.Linear(104,192),nn.ELU(),nn.Linear(192,192),nn.ELU(),nn.Linear(192,42)) for _ in range(COUNT)])
        for net in self.experts:nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    def forward(self,x):
        state=x[:,36:54].argmax(-1);features=torch.cat([x[:,:36],x[:,54:]],-1);out=x.new_zeros((len(x),42))
        for s,net in enumerate(self.experts):
            ids=torch.where(state==s)[0]
            if len(ids):out[ids]=net(features[ids])
        pose_active=~((state==0)|(state==15)|(state==16));hand_active=state!=0
        return torch.cat([out[:,:30]*pose_active[:,None],out[:,30:]*hand_active[:,None]],-1)
class ReleaseMemoryPolicy(IntentStatePolicy):
    def __init__(self):
        super().__init__();self.transition_net=ReleaseTransition();self.intent_net=ReleaseIntent();self.state_mean=torch.zeros(206);self.state_scale=torch.ones(206);self.intent_mean=torch.zeros(122);self.intent_scale=torch.ones(122)
    def state_features(self,a,initial,phase,omega,variation,previous):
        basic=CoordinationPolicy.features(self,a,initial,phase,omega,variation)
        return torch.cat([a,basic,torch.nn.functional.one_hot(previous,COUNT).to(a.dtype)],-1)
    def intent_features(self,a,initial,phase,omega,variation,state):
        basic=CoordinationPolicy.features(self,a,initial,phase,omega,variation)
        return torch.cat([basic,torch.nn.functional.one_hot(state,COUNT).to(a.dtype),a[:,102:120],a[:,14:26],a[:,66:78],a[:,140:152],a[:,:14]],-1)
class ReleaseMemoryModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True);state.setdefault('transition_net.log_transition_prior',torch.zeros(COUNT,COUNT));p=ReleaseMemoryPolicy();p.load_state_dict(state);p.eval();return cls(p)
