"""Learned conditional recurrent coordination; no runtime expert or stage input."""
import torch
from torch import nn
from .driving_intent_state_policy import IntentStatePolicy

class ConditionalTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(188,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,225))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():self.net[-1].bias.copy_((2*torch.eye(15)).reshape(-1))
    def forward(self,x):
        previous=x[:,-15:].argmax(-1)
        return self.net(x[:,:188]).reshape(-1,15,15)[torch.arange(len(x),device=x.device),previous]

class ConditionalIntent(nn.Module):
    def __init__(self,extra_dim=0):
        super().__init__()
        width=192 if extra_dim else 128
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(54+extra_dim,width),nn.ELU(),nn.Linear(width,width),nn.ELU(),nn.Linear(width,42)) for _ in range(15)])
        for net in self.experts:nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    def forward(self,x):
        state=x[:,36:51].argmax(-1);features=torch.cat([x[:,:36],x[:,51:]],-1)
        out=x.new_zeros((len(x),42))
        for s,net in enumerate(self.experts):
            ids=torch.where(state==s)[0]
            if len(ids):out[ids]=net(features[ids])
        active=(state!=0).to(x.dtype)[:,None]
        return torch.cat([out[:,:30]*active,out[:,30:]],-1)

class ConditionalMemoryPolicy(IntentStatePolicy):
    def __init__(self,full_feedback=False):
        super().__init__();self.transition_net=ConditionalTransition();self.intent_net=ConditionalIntent(50 if full_feedback else 0)
        if full_feedback:self.intent_mean=torch.zeros(119);self.intent_scale=torch.ones(119)
    def intent_features(self,a,initial,phase,omega,variation,state):
        x=super().intent_features(a,initial,phase,omega,variation,state)
        if self.intent_mean.numel()>69:x=torch.cat([x,a[:,14:26],a[:,66:78],a[:,140:152],a[:,:14]],-1)
        return x

class ConditionalMemoryModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True);p=ConditionalMemoryPolicy(full_feedback=state['intent_mean'].numel()>69);p.load_state_dict(state);p.eval();return cls(p)
