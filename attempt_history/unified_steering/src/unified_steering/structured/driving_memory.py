"""Supervised structured recurrent memory inside one learned24-action actor.

A learned linear recurrent cell integrates its angular latent. No expert phase,
role, stage or grasp target is supplied at execution. All actuator commands are
predicted by the joint neural decoder. Training labels include latent supervision.
"""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json
import numpy as np,torch
from torch import nn
from .driving_task import timed_reference

class MemoryPolicy(nn.Module):
    def __init__(self,base,local_hand=False):
        super().__init__();self.features_extractor=deepcopy(base.features_extractor)
        self.actor=deepcopy(base.mlp_extractor.actor);self.head=deepcopy(base.action_net)
        original=self.actor[0];expanded=nn.Linear(202,original.out_features)
        with torch.no_grad():expanded.weight.zero_();expanded.weight[:,:168]=original.weight;expanded.bias.copy_(original.bias)
        self.actor[0]=expanded;self.velocity_head=nn.Linear(7,1)
        self.register_buffer('memory_mean',torch.zeros(34));self.register_buffer('memory_scale',torch.ones(34))
        self.local_hand=local_hand
        if local_hand:self.hand_skip=nn.Linear(36,12)
        self.phase=0.;self.initial=None
    def forward(self,actor,privileged,memory):
        f=self.features_extractor(dict(actor=actor,privileged=privileged))[:,:168]
        result=self.head(self.actor(torch.cat([f,(memory-self.memory_mean)/self.memory_scale],-1)))
        if self.local_hand:
            local=torch.cat([actor[:,14:26],actor[:,66:78],memory[:,-12:]],-1)
            result=torch.cat([result[:,:12],self.hand_skip(local)+result[:,12:]],-1)
        return result
    def reset(self):self.phase=0.;self.initial=None
    def predict(self,observation,deterministic=True):
        a=torch.as_tensor(observation['actor'],dtype=torch.float32)
        if self.initial is None:self.initial=torch.cat([a[102:121],a[14:26]+.15*a[66:78]]).clone()
        with torch.no_grad():
            omega=self.velocity_head(a[122:129]).item()
            measured=a[120].item()-self.initial[18].item()
            self.phase=float(np.clip(self.phase+.05*omega,measured-np.pi/6,measured+np.pi/6))
            memory=torch.cat([torch.tensor([self.phase,np.sin(self.phase),np.cos(self.phase)],dtype=torch.float32),self.initial])
            action=self(a[None],torch.as_tensor(observation['privileged'],dtype=torch.float32)[None],memory[None])[0]
        return action.clamp(-1,1).numpy(),None

class MemoryModel:
    def __init__(self,policy):self.policy=policy
    def reset(self):self.policy.reset()
    def predict(self,*args,**kwargs):return self.policy.predict(*args,**kwargs)
    def save(self,path):
        from .driving_policy import DrivingFeatures
        torch.save(dict(state_dict=self.policy.state_dict(),local_hand=self.policy.local_hand,normalized=isinstance(self.policy.features_extractor,DrivingFeatures)),path)
    @classmethod
    def load(cls,path):
        import gymnasium as gym
        from .legacy.regrasp.policy import TeacherPolicy
        from .driving_policy import DrivingTeacherPolicy
        value=torch.load(path,map_location='cpu',weights_only=True)
        space=gym.spaces.Dict(actor=gym.spaces.Box(-np.inf,np.inf,(152,),np.float32),privileged=gym.spaces.Box(-np.inf,np.inf,(54,),np.float32))
        base=(DrivingTeacherPolicy if value['normalized'] else TeacherPolicy)(space,gym.spaces.Box(-1.,1.,(24,),np.float32),lambda _:1e-5)
        policy=MemoryPolicy(base,local_hand=value.get('local_hand',False));policy.load_state_dict(value['state_dict']);policy.eval();policy.reset();return cls(policy)

def memory_for(actor,omega):
    initial=np.r_[actor[0,102:121],actor[0,14:26]+.15*actor[0,66:78]]
    phase=0.;rows=[]
    for a,w in zip(actor,omega):
        measured=float(a[120]-actor[0,120]);phase=np.clip(phase+.05*w,measured-np.pi/6,measured+np.pi/6)
        rows.append(np.r_[phase,np.sin(phase),np.cos(phase),initial])
    return np.asarray(rows,np.float32)

def train(args):
    from .evaluate import load_teacher
    from .driving_learning import write
    torch.set_num_threads(1);p=json.loads(args.protocol.read_text());torch.manual_seed(p['seed']);rng=np.random.default_rng(p['seed'])
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'protocol.json',p)
    base=load_teacher(args.model);model=base if isinstance(base,MemoryModel) else MemoryModel(MemoryPolicy(base.policy));policy=model.policy
    episodes=[];sources=[];x=[];y=[]
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            z=np.load(path);a=z['actor'];case=json.loads((directory/'cases'/(path.stem+'.json')).read_text())
            angle,velocity,t=timed_reference(float(a[0,120]),np.deg2rad(case['amplitude_deg']),case['direction'],case['template'])
            ids=np.arange(len(a))+3;omega=velocity[np.minimum(ids,len(velocity)-1)]+6*(angle[np.minimum(ids,len(angle)-1)]-a[:,120])
            episodes.append({key:z[key] for key in ['actor','privileged','action']}|dict(omega=omega))
            x.append(a[:,122:129]);y.append(omega);sources.append(dict(path=str(path),samples=len(a)))
    design=np.c_[np.concatenate(x),np.ones(sum(len(a) for a in x))];target=np.concatenate(y)
    coefficient=np.linalg.lstsq(design,target,rcond=None)[0]
    with torch.no_grad():policy.velocity_head.weight.copy_(torch.as_tensor(coefficient[:7],dtype=torch.float32)[None]);policy.velocity_head.bias.fill_(coefficient[7])
    data={key:[] for key in ['actor','privileged','action','memory']};phase_errors=[]
    for e in episodes:
        prediction=np.c_[e['actor'][:,122:129],np.ones(len(e['actor']))]@coefficient
        memory=memory_for(e['actor'],prediction);truth=memory_for(e['actor'],e['omega']);phase_errors.extend(memory[:,0]-truth[:,0])
        for key in ['actor','privileged','action']:data[key].append(e[key])
        data['memory'].append(memory)
    z=np.load(p['retention_dataset']);actor=z['actor'];starts=np.flatnonzero(np.max(abs(actor[:,78:102]),axis=1)<1e-8)
    starts=np.unique(np.r_[0,starts,len(actor)])
    old_memory=np.zeros((len(actor),34),np.float32)
    for start,end in zip(starts[:-1],starts[1:]):
        prediction=np.c_[actor[start:end,122:129],np.ones(end-start)]@coefficient
        old_memory[start:end]=memory_for(actor[start:end],prediction)
    ids=rng.choice(len(actor),p.get('retention_samples',20000),replace=False)
    for key in ['actor','privileged','action']:data[key].append(z[key][ids])
    data['memory'].append(old_memory[ids]);data={k:torch.from_numpy(np.concatenate(v)) for k,v in data.items()}
    with torch.no_grad():
        mean=data['memory'].mean(0);scale=data['memory'].std(0).clamp_min(.02)
        first=policy.actor[0];weight=first.weight[:,168:].clone()
        first.bias.add_(weight@((mean-policy.memory_mean)/policy.memory_scale))
        first.weight[:,168:].mul_(scale/policy.memory_scale)
        policy.memory_mean.copy_(mean);policy.memory_scale.copy_(scale)
    metadata=dict(policy_variant='phase-memory-local-hand-v2' if policy.local_hand else 'phase-memory-driving-v1',interface=p['interface'],task=p['task'],training_method='latent-supervised-structured-memory-BC-DAgger',
        wrist_block_only=p.get('wrist_block_only',False),hand_branch_previously_fitted=policy.local_hand,parent=str(args.model),parent_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),status='training',critic_updated=False,script_at_evaluation=False,
        all24_mean_action_channels_trainable=True,velocity_training_rmse=float(np.sqrt(np.mean((design@coefficient-target)**2))),
        phase_training_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(phase_errors))))),retention_detected_episodes=len(starts)-1)
    write(args.output/'config.json',metadata);write(args.output/'data_sources.json',sources)
    parameters=list(policy.features_extractor.parameters())+list(policy.actor.parameters())+list(policy.head.parameters())
    if policy.local_hand and not p.get('wrist_block_only',False):parameters+=list(policy.hand_skip.parameters())
    optimizer=torch.optim.Adam(parameters,lr=p['bc_learning_rate']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=p['epochs'],eta_min=3e-7)
    losses=[];weights=torch.tensor([4.]*12+[1.]*12);policy.train();n=len(data['actor'])
    for epoch in range(p['epochs']):
        total=0.
        for ids in torch.randperm(n).split(p.get('bc_batch_size',1024)):
            prediction=policy(data['actor'][ids],data['privileged'][ids],data['memory'][ids])
            hold=torch.where(data['actor'][ids,128].abs()<1e-5,2.,1.)
            loss=(((prediction-data['action'][ids]).square()*(weights*torch.tensor([1.]*12+[0.]*12) if p.get('wrist_block_only',False) else weights)).mean(-1)*hold).mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();total+=loss.item()*len(ids)
        losses.append(total/n);scheduler.step()
        if epoch%10==0 or epoch==p['epochs']-1:print(json.dumps(dict(epoch=epoch,loss=losses[-1],samples=n)),flush=True)
    policy.eval();policy.reset();model.save(args.output/'teacher.zip');write(args.output/'losses.json',losses)
    metadata.update(status='trained_requires_development',final_loss=losses[-1],model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest())
    write(args.output/'config.json',metadata)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,nargs='+',required=True)
    # Checkpoint loaders import the canonical module. Avoid two class identities
    # (__main__.MemoryModel vs package.MemoryModel) when continuing training.
    from unified_steering.structured.driving_memory import train as canonical_train
    canonical_train(p.parse_args())
