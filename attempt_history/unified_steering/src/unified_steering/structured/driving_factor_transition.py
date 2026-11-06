"""Learn when to advance separately from where to advance.

Every destination retains positive neural probability. A self-memory skip is
an architectural persistence prior, not a prescribed allowed-transition list.
"""
from copy import deepcopy
import torch
from torch import nn
from .driving_contact_history import ContactHistoryPolicy
from .driving_history_intent import COUNT

class FactorTransition(nn.Module):
    def __init__(self,phase_net=None):
        super().__init__();self.register_buffer('log_prior',torch.zeros(COUNT,COUNT));self.register_buffer('shared_phase',torch.tensor(True))
        self.gates=nn.ModuleList([nn.Sequential(nn.Linear(235,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,2)) for _ in range(COUNT)])
        self.destinations=nn.ModuleList([nn.Sequential(nn.Linear(6,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,COUNT)) for _ in range(COUNT)])
        self.phase_net=deepcopy(phase_net) if phase_net is not None else nn.Sequential(nn.Linear(6,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,COUNT))
    def components(self,x):
        previous=x[:,-COUNT:].argmax(-1);gate=x.new_zeros((len(x),2));destination=x.new_zeros((len(x),COUNT));phase=x[:,[120,170,183,186,187,128]]
        for s in range(COUNT):
            ids=torch.where(previous==s)[0]
            if len(ids):gate[ids]=self.gates[s](x[ids,:235]);destination[ids]=self.destinations[s](phase[ids])
        return previous,gate,destination,self.phase_net(phase)
    def forward(self,x):
        previous,gate,destination,phase=self.components(x);prob=gate.softmax(-1);result=prob[:,1,None]*destination.softmax(-1)+prob[:,0,None]*torch.nn.functional.one_hot(previous,COUNT)
        initial=(previous==0)|(previous==15)|(previous==16)
        return torch.where(initial[:,None],phase,result.clamp_min(1e-30).log())
    def supervised_loss(self,x,label,event_weight):
        previous,gate,destination,phase=self.components(x);event=(label!=previous).long();initial=(previous==0)|(previous==15)|(previous==16)
        when=torch.nn.functional.cross_entropy(gate,event,reduction='none')*torch.where(event.bool(),event_weight,1.)
        where=torch.nn.functional.cross_entropy(destination,label,reduction='none')*event*event_weight
        initial_loss=torch.nn.functional.cross_entropy(phase,label,reduction='none')*torch.where(event.bool(),event_weight,1.)
        return torch.where(initial,initial_loss,when+where)
class FactorContactPolicy(ContactHistoryPolicy):
    def __init__(self):super().__init__();self.transition_net=FactorTransition()
class FactorContactModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        p=FactorContactPolicy();state=torch.load(path,map_location='cpu',weights_only=True);state.setdefault('bounded_intent',torch.tensor(False));p.load_state_dict(state);p.eval();return cls(p)
