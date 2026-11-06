"""Privileged training teacher: measured hand/rim force and contact torque.

Contact observations are simulation truth, deliberately excluded from any
claim of deployable proprioceptive student capability.
"""
from copy import deepcopy
import torch
from torch import nn
from .driving_history_intent import HistoryIntentPolicy,HistoryHead,COUNT

class ContactHead(HistoryHead):
    def __init__(self,transition=False,parent=None):
        super().__init__(transition)
        if parent is not None:self.load_state_dict(parent.state_dict())
        for network in self.networks:
            old=network[0];new=nn.Linear(235,192)
            with torch.no_grad():new.weight.zero_();new.weight[:,:231]=old.weight;new.bias.copy_(old.bias)
            network[0]=new
    def forward(self,x):
        state=x[:,-COUNT:].argmax(-1);result=x.new_zeros((len(x),COUNT if self.transition else 42))
        for s,net in enumerate(self.networks):
            ids=torch.where(state==s)[0]
            if len(ids):result[ids]=net(x[ids,:235])
        if self.transition:
            result=result+self.log_prior[state]
            if self.shared_phase:
                shared=self.phase_net(x[:,[120,170,183,186,187,128]])
                result=torch.where(((state==0)|(state==15)|(state==16))[:,None],shared,result)
            return result
        return torch.cat([result[:,:30]*((state!=0)&(state!=15)&(state!=16))[:,None],result[:,30:]*(state!=0)[:,None]],-1)

class ContactHistoryPolicy(HistoryIntentPolicy):
    def __init__(self,parent=None):
        super().__init__()
        if parent is not None:
            for name in ['base','output_scale','wrist_scale','correct_stage_age','anchored_hand','relative_feedback','bounded_intent']:setattr(self,name,deepcopy(getattr(parent,name)))
        self.intent_net=ContactHead(parent=parent.intent_net if parent is not None else None)
        self.transition_net=ContactHead(True,parent.transition_net if parent is not None else None)
        self.feature_mean=torch.cat([parent.feature_mean[:231],torch.zeros(4),parent.feature_mean[231:]]) if parent is not None else torch.zeros(267)
        self.feature_scale=torch.cat([parent.feature_scale[:231],torch.ones(4),parent.feature_scale[231:]]) if parent is not None else torch.ones(267)
    def reset(self):super().reset();self.contact=None
    def features_history(self,*args):
        x=super().features_history(*args)
        if self.contact is None or len(self.contact)!=len(x):raise ValueError('Privileged teacher requires matched force/torque observations')
        return torch.cat([x[:,:231],self.contact.to(x),x[:,231:]],-1)
    def predict(self,observation,deterministic=True):
        if 'teacher_contact' not in observation:raise ValueError('Missing privileged teacher contact observation; deployment student is a separate stage')
        self.contact=torch.as_tensor(observation['teacher_contact'],dtype=torch.float32).reshape(1,4)
        return super().predict(observation,deterministic)
class ContactHistoryModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        p=ContactHistoryPolicy();state=torch.load(path,map_location='cpu',weights_only=True);state.setdefault('bounded_intent',torch.tensor(False));p.load_state_dict(state);p.eval();return cls(p)
