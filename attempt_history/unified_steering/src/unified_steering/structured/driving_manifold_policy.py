"""Learned contact-manifold bottleneck; all runtime decisions remain neural."""
import numpy as np,torch
from torch import nn
from .driving_intent_state_policy import IntentStatePolicy
from .driving_pose_policy import rotation6,exp_rotation,log_rotation

class ManifoldPolicy(IntentStatePolicy):
    def __init__(self):
        super().__init__()
        self.intent_net=nn.Sequential(nn.Linear(69,384),nn.ELU(),nn.Linear(384,384),nn.ELU(),nn.Linear(384,256),nn.ELU(),nn.Linear(256,28))
        nn.init.zeros_(self.intent_net[-1].weight);nn.init.zeros_(self.intent_net[-1].bias)
        self.register_buffer('initial_geometry_anchor',torch.tensor(0.))
        self.register_buffer('parameter_scale',torch.tensor([np.pi,np.pi,.1,.05,.2,.2,.2,1.]*2+[.5]*12,dtype=torch.float32))
    def baseline_parameters(self,a,initial,phase):
        b=torch.zeros((len(a),28),device=a.device,dtype=a.dtype)
        for side in range(2):b[:,side*8]=phase;b[:,side*8+1]=phase;b[:,side*8+7]=1.
        b[:,16:]=self.base_intent(a,initial,phase)[:,30:]
        return b
    def decode_parameters(self,param,initial):
        poses=[]
        for side in range(2):
            x=param[:,side*8:(side+1)*8]
            position_source=self.base.pose(x[:,0],initial)[:,side];rotation_source=self.base.pose(x[:,1],initial)[:,side]
            position=position_source[:,:3].clone();position[:,0]=position[:,0]+x[:,2]
            radial=position[:,1:]/torch.linalg.vector_norm(position[:,1:],dim=-1,keepdim=True).clamp_min(.05)
            position[:,1:]=position[:,1:]+x[:,3:4]*radial
            rotation=exp_rotation(x[:,4:7])@rotation6(rotation_source[:,3:9])
            poses.append(torch.cat([position,rotation[:,:,:2].reshape(-1,6),position_source[:,9:12]*x[:,7:8],rotation_source[:,12:15]*x[:,7:8]],-1))
        return torch.cat(poses+[param[:,16:]],-1)
    def intention_for_state(self,a,initial,phase,omega,variation,state):
        basic=super(IntentStatePolicy,self).features(a,initial,phase,omega,variation)
        x=torch.cat([basic,torch.nn.functional.one_hot(state,15).to(a.dtype),a[:,102:120]],-1)
        param=self.baseline_parameters(a,initial,phase)+self.intent_net((x-self.intent_mean)/self.intent_scale)*self.parameter_scale
        active=(1-self.initial_geometry_anchor*(state==0).to(a.dtype))[:,None]
        baseline=self.baseline_parameters(a,initial,phase)
        param=torch.cat([baseline[:,:16]+active*(param[:,:16]-baseline[:,:16]),param[:,16:]],-1)
        return self.decode_parameters(param,initial)
    def encode_labels(self,a,initial,phase,target):
        params=self.baseline_parameters(a,initial,phase).clone();pose=target[:,:30].reshape(-1,2,15)
        init=initial[:,:18].reshape(-1,2,9);r0=rotation6(init[:,:,3:9])
        for side in range(2):
            phi=torch.atan2(pose[:,side,2],pose[:,side,1])-torch.atan2(init[:,side,2],init[:,side,1])
            phi=torch.atan2(phi.sin(),phi.cos());desired=rotation6(pose[:,side,3:9]);relative=desired@r0[:,side].transpose(-1,-2)
            rophi=torch.atan2(relative[:,2,1],relative[:,1,1]);basepos=self.base.pose(phi,initial)[:,side];baserot=self.base.pose(rophi,initial)[:,side]
            params[:,side*8]=phi;params[:,side*8+1]=rophi
            params[:,side*8+2]=pose[:,side,0]-basepos[:,0]
            params[:,side*8+3]=torch.linalg.vector_norm(pose[:,side,1:3],dim=-1)-torch.linalg.vector_norm(basepos[:,1:3],dim=-1)
            params[:,side*8+4:side*8+7]=log_rotation(desired@rotation6(baserot[:,3:9]).transpose(-1,-2))
            params[:,side*8+7]=(torch.linalg.vector_norm(pose[:,side,9:15],dim=-1)>1e-5).to(a.dtype)
        params[:,16:]=target[:,30:]
        return params

class ManifoldModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        p=ManifoldPolicy();state=torch.load(path,map_location='cpu',weights_only=True);state.setdefault('initial_geometry_anchor',torch.tensor(0.));p.load_state_dict(state);p.eval();return cls(p)
