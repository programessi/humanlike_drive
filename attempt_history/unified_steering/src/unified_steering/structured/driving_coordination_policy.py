"""Learned wrist/hand intentions with proprioceptive feedback in one actor.

The training-only expert supplies pose/hand labels. Inference loads weights,
proprioception and finite command history; no expert, stage or geometry table.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .driving_pose_policy import PosePolicy, rotation6,exp_rotation,log_rotation

class CoordinationPolicy(nn.Module):
    input_dim=36
    def __init__(self,input_dim=36):
        super().__init__();self.input_dim=input_dim;self.base=PosePolicy()
        self.intent_net=nn.Sequential(nn.Linear(self.input_dim,384),nn.ELU(),nn.Linear(384,384),nn.ELU(),nn.Linear(384,256),nn.ELU(),nn.Linear(256,42))
        nn.init.zeros_(self.intent_net[-1].weight);nn.init.zeros_(self.intent_net[-1].bias)
        self.register_buffer('input_mean',torch.zeros(self.input_dim));self.register_buffer('input_scale',torch.ones(self.input_dim))
        self.register_buffer('output_scale',torch.tensor(([.05]*3+[.4]*6+[.2]*3+[1.]*3)*2+[.5]*12))
        self.hand_gain=nn.Parameter(torch.tensor(.1));self.register_buffer('wrist_scale',torch.tensor([.1,np.deg2rad(60.)]))
        self.reset()
    def reset(self):self.phase=0.;self.variation=0.;self.initial=None
    def features(self,a,initial,phase,omega,variation):
        context=torch.cat([initial,phase[:,None],phase.sin()[:,None],phase.cos()[:,None],omega[:,None],variation[:,None]],-1)
        return torch.cat([a,context],-1) if self.input_dim==188 else context
    def base_intent(self,a,initial,phase):
        pose=self.base.pose(phase,initial).reshape(-1,30)
        hand_action=self.base.hand_skip(torch.cat([a[:,14:26],a[:,66:78],initial[:,-12:]],-1))
        measured=a[:,14:26];current=measured+.15*a[:,66:78]
        hand=current+(.03*hand_action-.005*(current-measured))/.1
        return torch.cat([pose,hand],-1)
    def intention(self,a,initial,phase,omega,variation):
        x=self.features(a,initial,phase,omega,variation)
        return self.base_intent(a,initial,phase)+self.intent_net((x-self.input_mean)/self.input_scale)*self.output_scale
    def action_from_intent(self,a,intent,omega):
        pose=intent[:,:30].reshape(-1,2,15);fk=a[:,102:120].reshape(-1,2,9);gap=a[:,140:152].reshape(-1,2,6)
        current_position=fk[:,:,:3]*.2+gap[:,:,:3]*.02
        current_rotation=exp_rotation(gap[:,:,3:]*.2)@rotation6(fk[:,:,3:])
        linear=self.base.position_gain*(pose[:,:,:3]-current_position)+pose[:,:,9:12]*omega[:,None,None]
        angular=self.base.orientation_gain*log_rotation(rotation6(pose[:,:,3:9])@current_rotation.transpose(-1,-2))+pose[:,:,12:15]*omega[:,None,None]
        radius=torch.linalg.vector_norm(current_position[:,:,1:],dim=-1).clamp_min(.085)
        cosine=current_position[:,:,1]/radius;sine=current_position[:,:,2]/radius
        wrist=torch.cat([torch.stack([-linear[:,:,1]*sine+linear[:,:,2]*cosine,linear[:,:,1]*cosine+linear[:,:,2]*sine,linear[:,:,0]],-1)/self.wrist_scale[0],angular/self.wrist_scale[1]],-1).reshape(-1,12)
        measured=a[:,14:26];current=measured+.15*a[:,66:78]
        hand=(self.hand_gain*(intent[:,30:]-current)+.005*(current-measured))/.03
        return torch.cat([wrist,hand],-1)
    def predict(self,observation,deterministic=True):
        a=torch.as_tensor(observation['actor'],dtype=torch.float32)
        if self.initial is None:self.initial=torch.cat([a[102:121],a[14:26]+.15*a[66:78]]).clone()
        with torch.no_grad():
            omega=self.base.velocity_head(a[122:129]).item();measured=a[120].item()-self.initial[18].item();old=self.phase
            self.phase=float(np.clip(self.phase+.05*omega,measured-np.pi/6,measured+np.pi/6));self.variation+=abs(self.phase-old)
            p=torch.tensor([self.phase]);w=torch.tensor([omega]);v=torch.tensor([self.variation])
            intent=self.intention(a[None],self.initial[None],p,w,v)
            action=self.action_from_intent(a[None],intent,w)[0]
        return action.clamp(-1,1).numpy(),None

class CoordinationModel:
    def __init__(self,policy):self.policy=policy
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True);p=CoordinationPolicy(state['input_mean'].numel());p.load_state_dict(state);p.eval();return cls(p)

def states(a,p):
    initial=np.tile(np.r_[a[0,102:121],a[0,14:26]+.15*a[0,66:78]],(len(a),1)).astype(np.float32)
    with torch.no_grad():omega=p.base.velocity_head(torch.tensor(a[:,122:129])).numpy()[:,0]
    phase=[];variation=[];value=0.;distance=0.
    for row,w in zip(a,omega):
        before=value;measured=row[120]-initial[0,18];value=float(np.clip(value+.05*w,measured-np.pi/6,measured+np.pi/6));distance+=abs(value-before)
        phase.append(value);variation.append(distance)
    return initial,np.array(phase,np.float32),omega.astype(np.float32),np.array(variation,np.float32)

def train(args):
    from .evaluate import load_teacher
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    args.output.mkdir(exist_ok=False,parents=True)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['driving_coordination_policy.py','driving_pose_policy.py','driving_task.py','evaluate.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    parent=load_teacher(args.model).policy
    p=deepcopy(parent) if isinstance(parent,CoordinationPolicy) and parent.input_dim==args.feature_dim else CoordinationPolicy(args.feature_dim)
    if not isinstance(parent,CoordinationPolicy) or parent.input_dim!=args.feature_dim:p.base=deepcopy(parent.base if isinstance(parent,CoordinationPolicy) else parent)
    pieces={k:[] for k in ['a','initial','phase','omega','variation','target','action']};sources=[]
    for directory in args.data+args.retention:
        for path in sorted((directory/'labels').glob('*.npz')):
            case=directory/'cases'/f'{path.stem}.json'
            retaining=directory in args.retention
            if not retaining and case.exists() and not json.loads(case.read_text())['passed']:continue
            z=np.load(path);a=z['actor'].astype(np.float32);initial,phase,omega,variation=states(a,p)
            if retaining:
                with torch.no_grad():target=p.base_intent(torch.tensor(a),torch.tensor(initial),torch.tensor(phase)).numpy()
                ids=np.arange(0,len(a),max(1,args.retention_stride));ids=ids[np.abs(phase[ids])<np.deg2rad(15.)]
            else:
                target=np.c_[z['pose'],z['hand_target']].astype(np.float32);ids=np.arange(len(a))
            for key,values in dict(a=a,initial=initial,phase=phase,omega=omega,variation=variation,target=target,action=z['action']).items():pieces[key].append(values[ids])
            sources.append(dict(path=str(path),samples=len(ids),retention=retaining,expert_passed=True if not retaining else None))
    data={k:np.concatenate(v) for k,v in pieces.items()};n=len(data['a']);device='cuda' if torch.cuda.is_available() else 'cpu';p=p.to(device)
    t={k:torch.tensor(v,device=device) for k,v in data.items()}
    with torch.no_grad():
        x=p.features(t['a'],t['initial'],t['phase'],t['omega'],t['variation']);p.input_mean.copy_(x.mean(0));p.input_scale.copy_(x.std(0).clamp_min(.03))
        target=(t['target']-p.base_intent(t['a'],t['initial'],t['phase']))/p.output_scale
    # The residual learns all wrists and hands; no channel is masked at inference.
    optimizer=torch.optim.Adam(p.intent_net.parameters(),lr=args.lr);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.03)
    weight=torch.tensor(([4.]*3+[1.]*6+[.2]*3+[.1]*3)*2+[2.]*12,device=device);losses=[]
    writer=None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(args.output/'tensorboard'))
    except ImportError:pass
    for epoch in range(args.epochs):
        total=0.
        for ids in torch.randperm(n,device=device).split(1024):
            features=(x[ids]-p.input_mean)/p.input_scale
            if args.feature_dim==188 and args.augmentation>0:
                physical_std=torch.tensor([.04]*26+[.01]*26+[.1]*26+[.05]*24+[.05]*18+[0.]*20+[.05]*12+[0.]*36,device=device)
                features=features+args.augmentation*torch.randn_like(features)*physical_std/p.input_scale
            pred=p.intent_net(features);loss=((pred-target[ids]).square()*weight).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.intent_net.parameters(),5.);optimizer.step();total+=loss.item()*len(ids)
        losses.append(total/n);scheduler.step()
        if writer:writer.add_scalar('train/intent_loss',losses[-1],epoch)
        if epoch%20==0 or epoch==args.epochs-1:print(json.dumps(dict(epoch=epoch,loss=losses[-1],samples=n,device=device)),flush=True)
    if writer:writer.close()
    p=p.cpu().eval();CoordinationModel(p).save(args.output/'teacher.zip')
    config=dict(policy_variant='coordination-pose-driving-v1',parent=str(args.model),training_method='BC-intent-distillation-with-policy-local-feedback',actor_dim=152,privileged_used=False,action_dim=24,intent_network=[p.input_dim,384,384,256,42],expert_at_execution=False,stage_at_execution=False,critic_updated=False,epochs=args.epochs,seed=args.seed,samples=n,model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest(),feature_dim=p.input_dim,augmentation=args.augmentation,status='requires_physical_development',task=json.loads((args.data[0]/'metadata.json').read_text())['task'])
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');(args.output/'data_sources.json').write_text(json.dumps(sources,indent=2)+'\n');(args.output/'losses.json').write_text(json.dumps(losses)+'\n');print(json.dumps(config),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--data',type=Path,nargs='+',required=True);a.add_argument('--retention',type=Path,nargs='*',default=[]);a.add_argument('--retention-stride',type=int,default=4);a.add_argument('--epochs',type=int,default=400);a.add_argument('--lr',type=float,default=3e-4);a.add_argument('--seed',type=int,default=650);a.add_argument('--feature-dim',type=int,choices=[36,188],default=36);a.add_argument('--augmentation',type=float,default=1.)
    from unified_steering.structured.driving_coordination_policy import train as canonical_train
    canonical_train(a.parse_args())
