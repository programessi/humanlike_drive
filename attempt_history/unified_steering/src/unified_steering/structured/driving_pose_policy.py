"""Learned pose bottleneck and local feedback inside one24-output policy.

The pose MLP is distilled from labelled geometry during training. Inference
loads only weights, proprioception and command history; no expert/table/stage
is instantiated. Feedback algebra is differentiable and is part of the actor.
"""
from pathlib import Path
import argparse,json,hashlib
from copy import deepcopy
import numpy as np,torch
from torch import nn


def rotation6(x):
    pair=x.reshape(*x.shape[:-1],3,2)
    u=torch.nn.functional.normalize(pair[...,0],dim=-1)
    v=pair[...,1]-(pair[...,1]*u).sum(-1,keepdim=True)*u
    v=torch.nn.functional.normalize(v,dim=-1)
    return torch.stack([u,v,torch.cross(u,v,dim=-1)],-1)


def exp_rotation(v):
    a=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    k=torch.stack([o,-z,y,z,o,-x,-y,x,o],-1).reshape(*v.shape[:-1],3,3)
    eye=torch.eye(3,dtype=v.dtype,device=v.device).expand_as(k)
    f=torch.sinc(a/torch.pi)[...,None]
    g=(.5*torch.sinc(a/(2*torch.pi)).square())[...,None]
    return eye+f*k+g*(k@k)


def log_rotation(r):
    v=.5*torch.stack([r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]],-1)
    sine=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    cosine=((r.diagonal(dim1=-2,dim2=-1).sum(-1,keepdim=True)-1)*.5).clamp(-1.,1.)
    return v*(torch.atan2(sine,cosine)/sine.clamp_min(1e-7))


class PosePolicy(nn.Module):
    def __init__(self):
        super().__init__();self.pose_net=nn.Sequential(nn.Linear(21,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,30))
        self.velocity_head=nn.Linear(7,1);self.hand_skip=nn.Linear(36,12)
        self.position_gain=nn.Parameter(torch.tensor(3.));self.orientation_gain=nn.Parameter(torch.tensor(3.))
        self.register_buffer('input_mean',torch.zeros(21));self.register_buffer('input_scale',torch.ones(21))
        self.register_buffer('output_mean',torch.zeros(30));self.register_buffer('output_scale',torch.ones(30))
        self.phase=0.;self.initial=None
    def pose(self,phase,initial):
        x=torch.cat([phase[:,None],phase.sin()[:,None],phase.cos()[:,None],initial[:,:18]],-1)
        return (self.pose_net((x-self.input_mean)/self.input_scale)*self.output_scale+self.output_mean).reshape(-1,2,15)
    def action(self,a,initial,phase,omega):
        pose=self.pose(phase,initial);fk=a[:,102:120].reshape(-1,2,9);gap=a[:,140:152].reshape(-1,2,6)
        current_position=fk[:,:,:3]*.2+gap[:,:,:3]*.02
        current_rotation=exp_rotation(gap[:,:,3:]*.2)@rotation6(fk[:,:,3:])
        desired_rotation=rotation6(pose[:,:,3:9])
        linear=self.position_gain*(pose[:,:,:3]-current_position)+pose[:,:,9:12]*omega[:,None,None]
        angular=self.orientation_gain*log_rotation(desired_rotation@current_rotation.transpose(-1,-2))+pose[:,:,12:15]*omega[:,None,None]
        radius=torch.linalg.vector_norm(current_position[:,:,1:],dim=-1).clamp_min(.085)
        cosine=current_position[:,:,1]/radius;sine=current_position[:,:,2]/radius
        wrist=torch.cat([torch.stack([-linear[:,:,1]*sine+linear[:,:,2]*cosine,linear[:,:,1]*cosine+linear[:,:,2]*sine,linear[:,:,0]],-1)/.05,angular/np.deg2rad(40.)],-1).reshape(-1,12)
        hand=self.hand_skip(torch.cat([a[:,14:26],a[:,66:78],initial[:,-12:]],-1))
        return torch.cat([wrist,hand],-1)
    def reset(self):self.phase=0.;self.initial=None
    def predict(self,observation,deterministic=True):
        a=torch.as_tensor(observation['actor'],dtype=torch.float32)
        if self.initial is None:self.initial=torch.cat([a[102:121],a[14:26]+.15*a[66:78]]).clone()
        with torch.no_grad():
            omega=self.velocity_head(a[122:129]).item();measured=a[120].item()-self.initial[18].item()
            self.phase=float(np.clip(self.phase+.05*omega,measured-np.pi/6,measured+np.pi/6))
            result=self.action(a[None],self.initial[None],torch.tensor([self.phase]),torch.tensor([omega]))[0]
        return result.clamp(-1,1).numpy(),None


class PoseModel:
    def __init__(self,policy):self.policy=policy
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        p=PosePolicy();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));p.eval();return cls(p)


def geometry_labels(phase,initial,table):
    from scipy.spatial.transform import Rotation
    count=len(phase);p0=initial[:,:18].reshape(-1,2,9)[:,:,:3]*.2
    pair=initial[:,:18].reshape(-1,2,9)[:,:,3:].reshape(-1,2,3,2)
    r0=np.concatenate([pair,np.cross(pair[:,:,:,0],pair[:,:,:,1])[:,:,:,None]],-1)
    rows=json.loads(Path(table).read_text())['cases'];grid=np.array(sorted({r['rotation_from_prepared_grip_deg'] for r in rows}),float)
    labels=[]
    for side in range(2):
        values=[next(r for r in rows if r['rotation_from_prepared_grip_deg']==g)['left' if side==0 else 'right']['tangent_roll_deg'] for g in grid]
        radial=p0[:,side].copy();radial[:,0]=0.;radial/=np.linalg.norm(radial,axis=-1,keepdims=True)
        rim=.14*radial;tangent=np.cross(np.tile([1.,0.,0.],(count,1)),radial)*(1 if side==0 else -1)
        def get(angle):
            psi=np.deg2rad(np.interp(np.rad2deg(angle),grid,values));turn=Rotation.from_rotvec(np.c_[angle,np.zeros((count,2))]).as_matrix();roll=Rotation.from_rotvec(tangent*psi[:,None]).as_matrix()
            pos=np.einsum('nij,nj->ni',turn,rim+np.einsum('nij,nj->ni',roll,p0[:,side]-rim))
            rot=turn@roll@r0[:,side]
            free=p0[:,side,0]<-.06;pos[free]=p0[free,side];rot[free]=r0[free,side]
            return pos,rot
        p,r=get(phase);pn,rn=get(phase+1e-4);dp=(pn-p)/1e-4;dr=Rotation.from_matrix(rn@r.transpose(0,2,1)).as_rotvec()/1e-4
        labels.append(np.c_[p,r[:,:,:2].reshape(count,6),dp,dr])
    return np.stack(labels,1).reshape(count,30).astype(np.float32)


def train(args):
    from .evaluate import load_teacher
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    args.output.mkdir(exist_ok=False,parents=True);base=load_teacher(args.model).policy;p=deepcopy(base) if hasattr(base,'pose_net') else PosePolicy()
    resumed=hasattr(base,'pose_net')
    p.velocity_head=deepcopy(base.velocity_head);p.hand_skip=deepcopy(base.hand_skip)
    initials=[];sources=[]
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            a=np.load(path)['actor'];initials.append(np.r_[a[0,102:121],a[0,14:26]+.15*a[0,66:78]]);sources.append(str(path))
    # Old screened reset states are training data, not new evaluation data.
    z=np.load(args.retention)['actor'];starts=np.flatnonzero(np.max(abs(z[:,78:102]),axis=1)<1e-8)
    for a in z[starts]:initials.append(np.r_[a[102:121],a[14:26]+.15*a[66:78]])
    initials=np.array(initials,np.float32)
    n=100000;initial=initials[rng.integers(len(initials),size=n)].copy();phase=rng.uniform(-125,125,n)*np.pi/180
    # Geometry-only augmentation, not a claim of additional physical success.
    initial[:,:18].reshape(-1,2,9)[:,:,:3]+=rng.normal(0,.002,(n,2,3))
    labels=geometry_labels(phase,initial,args.table)
    x=np.c_[phase,np.sin(phase),np.cos(phase),initial[:,:18]].astype(np.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu';p=p.to(device)
    with torch.no_grad():
        if not resumed:p.input_mean.copy_(torch.tensor(x[:-2000].mean(0),device=device));p.input_scale.copy_(torch.tensor(x[:-2000].std(0),device=device).clamp_min(.02))
        if not resumed:p.output_mean.copy_(torch.tensor(labels[:-2000].mean(0),device=device));p.output_scale.copy_(torch.tensor(labels[:-2000].std(0),device=device).clamp_min(.03))
    x=torch.tensor(x,device=device);y=torch.tensor(labels,device=device);optimizer=torch.optim.Adam(p.pose_net.parameters(),lr=(3e-5 if resumed else 3e-4))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=3e-6);losses=[]
    for epoch in range(args.epochs):
        total=0.
        for ids in torch.randperm(n-2000,device=device).split(2048):
            out=p.pose_net((x[ids]-p.input_mean)/p.input_scale);target=(y[ids]-p.output_mean)/p.output_scale
            weight=torch.tensor(([10.]*3+[2.]*6+[.5]*3+[.25]*3)*2,device=device) if resumed else 1.
            loss=((out-target).square()*weight).mean();optimizer.zero_grad();loss.backward();optimizer.step();total+=loss.item()*len(ids)
        losses.append(total/(n-2000));scheduler.step()
        if epoch%10==0 or epoch==args.epochs-1:print(json.dumps(dict(epoch=epoch,loss=losses[-1],device=device)),flush=True)
    p=p.cpu().eval();model=PoseModel(p);model.save(args.output/'teacher.zip')
    with torch.no_grad():
        pred=(p.pose_net((x[-2000:].cpu()-p.input_mean)/p.input_scale)*p.output_scale+p.output_mean).reshape(-1,2,15)
        truth=y[-2000:].cpu().reshape(-1,2,15);pos=torch.linalg.vector_norm(pred[:,:,:3]-truth[:,:,:3],dim=-1).numpy()*1000
        angle=torch.linalg.vector_norm(log_rotation(rotation6(pred[:,:,3:9])@rotation6(truth[:,:,3:9]).transpose(-1,-2)),dim=-1).numpy()*180/np.pi
    config=dict(policy_variant='pose-bottleneck-driving-v1',parent=str(args.model),model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest(),training_method='supervised-geometry-pose-bottleneck-with-policy-local-feedback',all24_mean_action_channels_trainable=True,script_at_evaluation=False,critic_updated=False,geometry_augmentation_samples=n-2000,geometry_validation_samples=2000,physical_training_sources=sources,table=str(args.table),table_sha256=hashlib.sha256(args.table.read_bytes()).hexdigest(),pose_validation_position_max_mm=float(pos.max()),pose_validation_orientation_max_deg=float(angle.max()),epochs=args.epochs,seed=args.seed,status='requires_physical_development',task=json.loads(args.model.with_name('config.json').read_text())['task'])
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');(args.output/'losses.json').write_text(json.dumps(losses)+'\n');print(json.dumps(config),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--data',type=Path,nargs='+',required=True);a.add_argument('--table',type=Path,default=Path('results/driving_r0_geometry_roll_v1/summary.json'));a.add_argument('--retention',type=Path,default=Path('runs/r23_screened_bootstrap_v4/dataset.npz'));a.add_argument('--epochs',type=int,default=400);a.add_argument('--seed',type=int,default=640)
    from unified_steering.structured.driving_pose_policy import train as canonical_train
    canonical_train(a.parse_args())
