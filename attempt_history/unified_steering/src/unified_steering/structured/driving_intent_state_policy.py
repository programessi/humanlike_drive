"""Neural categorical memory for one coordinated wrist/hand actor.

Previous categorical memory is predicted by neural weights. Expert stage labels
are training supervision only: no transition mask, wheel-angle trigger, expert
or stage command exists at execution.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib,shutil
import numpy as np,torch
from torch import nn
from .driving_coordination_policy import CoordinationPolicy,states

STAGES=['clear','relocate','ready','approach','insert','close','unload']
class IntentStatePolicy(CoordinationPolicy):
    def __init__(self):
        super().__init__(36)
        self.intent_net=nn.Sequential(nn.Linear(69,384),nn.ELU(),nn.Linear(384,384),nn.ELU(),nn.Linear(384,256),nn.ELU(),nn.Linear(256,42))
        self.transition_net=nn.Sequential(nn.Linear(203,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,15))
        self.register_buffer('state_mean',torch.zeros(203));self.register_buffer('state_scale',torch.ones(203))
        self.register_buffer('intent_mean',torch.zeros(69));self.register_buffer('intent_scale',torch.ones(69))
        nn.init.zeros_(self.intent_net[-1].weight);nn.init.zeros_(self.intent_net[-1].bias)
    def reset(self):super().reset();self.memory=0
    def state_features(self,a,initial,phase,omega,variation,previous):
        basic=super().features(a,initial,phase,omega,variation)
        return torch.cat([a,basic,torch.nn.functional.one_hot(previous,15).to(a.dtype)],-1)
    def intent_features(self,a,initial,phase,omega,variation,state):
        basic=super().features(a,initial,phase,omega,variation)
        return torch.cat([basic,torch.nn.functional.one_hot(state,15).to(a.dtype),a[:,102:120]],-1)
    def intention_for_state(self,a,initial,phase,omega,variation,state):
        x=self.intent_features(a,initial,phase,omega,variation,state)
        return self.base_intent(a,initial,phase)+self.intent_net((x-self.intent_mean)/self.intent_scale)*self.output_scale
    def predict(self,observation,deterministic=True):
        a=torch.as_tensor(observation['actor'],dtype=torch.float32)
        if self.initial is None:self.initial=torch.cat([a[102:121],a[14:26]+.15*a[66:78]]).clone()
        with torch.no_grad():
            omega=self.base.velocity_head(a[122:129]).item();measured=a[120].item()-self.initial[18].item();old=self.phase
            self.phase=float(np.clip(self.phase+.05*omega,measured-np.pi/6,measured+np.pi/6));self.variation+=abs(self.phase-old)
            p=torch.tensor([self.phase]);w=torch.tensor([omega]);v=torch.tensor([self.variation]);initial=self.initial[None]
            x=self.state_features(a[None],initial,p,w,v,torch.tensor([self.memory]))
            self.memory=int(self.transition_net((x-self.state_mean)/self.state_scale).argmax(-1).item())
            intent=self.intention_for_state(a[None],initial,p,w,v,torch.tensor([self.memory]));action=self.action_from_intent(a[None],intent,w)[0]
        return action.clamp(-1,1).numpy(),None

class IntentStateModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        p=IntentStatePolicy();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));p.eval();return cls(p)

def stage_labels(directory,path,n,release=False):
    record=json.loads((directory/'cases'/f'{path.stem}.json').read_text());z=np.load(directory/'traces'/f'{path.stem}.npz');times=z['time']-.05
    actor=np.load(directory/'labels'/f'{path.stem}.npz')['actor'] if release else None
    opening_start=json.loads((directory/'protocol.json').read_text()).get('shuffle_parameters',{}).get('release_start_deg',5.) if release else None
    events=[ev for ev in record['expert_events'] if ev['event'].startswith('free_')];side=0 if record['direction']<0 else 1;cursor=0;labels=[]
    for t,stage in zip(times,z['expert_stage']):
        while cursor<len(events) and events[cursor]['time']<=t+1e-7:side=events[cursor]['side'];cursor+=1
        if stage=='initial' and release:
            i=len(labels);turn=-1 if actor[i,128]<0 else 1
            opened=abs(actor[i,120]-actor[0,120])>opening_start*3.141592653589793/180
            labels.append((15+(0 if turn<0 else 1)) if opened else 0)
        elif stage=='dual':labels.append(17 if release else 1+side*7+STAGES.index('close'))
        else:labels.append(0 if stage=='initial' else 1+side*7+STAGES.index(str(stage)))
    labels=np.array(labels,np.int64);return labels,np.r_[0,labels[:-1]]

def train(args):
    from .evaluate import load_teacher
    torch.set_num_threads(1);torch.manual_seed(args.seed);args.output.mkdir(exist_ok=False,parents=True)
    if args.release_memory:
        from .driving_release_memory import ReleaseMemoryPolicy,ReleaseMemoryModel
        p=ReleaseMemoryPolicy()
    elif args.conditional:
        from .driving_conditional_memory import ConditionalMemoryPolicy,ConditionalMemoryModel
        p=ConditionalMemoryPolicy(full_feedback=args.full_feedback)
    elif args.manifold:
        from .driving_manifold_policy import ManifoldPolicy,ManifoldModel
        p=ManifoldPolicy()
    else:p=IntentStatePolicy()
    parent=load_teacher(args.model).policy;p.base=deepcopy(parent.base if hasattr(parent,'base') else parent)
    if type(parent) is type(p) and parent.intent_mean.numel()==p.intent_mean.numel():p=deepcopy(parent)
    elif type(parent) is type(p):p.transition_net=deepcopy(parent.transition_net)
    elif hasattr(parent,"transition_net") and not args.conditional and not args.release_memory:p.transition_net=deepcopy(parent.transition_net)
    if args.manifold and args.anchor_initial:p.initial_geometry_anchor.fill_(1.)
    pieces={k:[] for k in ['a','initial','phase','omega','variation','target','action','state','previous','sample_weight']};sources=[]
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            record=json.loads((directory/'cases'/f'{path.stem}.json').read_text())
            safe_complete=record['complete'] and not record['hard_failure']
            if not record['passed'] and not (args.safe_complete and safe_complete) and not args.prefixes and directory not in args.prefix_data:continue
            z=np.load(path);keep=len(z['actor'])-(20 if not safe_complete else 0);
            if keep<40:continue
            a=z['actor'][:keep].astype(np.float32);initial,phase,omega,variation=states(a,p);state,previous=stage_labels(directory,path,len(a),release=args.release_memory);state=state[:keep];previous=previous[:keep]
            corrective=directory in args.prefix_data
            if corrective and args.network_previous:
                trace=np.load(directory/'traces'/f'{path.stem}.npz');previous=np.r_[0,trace['learned_intent_memory'][:-1]][:keep].astype(np.int64)
                if (previous<0).any() or (previous>=(18 if args.release_memory else 15)).any():raise ValueError('incompatible previous-memory labels')
            for k,v in dict(a=a,initial=initial,phase=phase,omega=omega,variation=variation,target=np.c_[z['pose'][:keep],z['hand_target'][:keep]].astype(np.float32),action=z['action'][:keep].astype(np.float32),state=state,previous=previous,sample_weight=np.full(len(a),args.corrective_weight if corrective else 1.,np.float32)).items():pieces[k].append(v)
            sources.append(dict(path=str(path),samples=len(a),expert_passed=record['passed'],safe_complete=safe_complete,corrective=corrective,training_weight=args.corrective_weight if corrective else 1.,network_previous_used=corrective and args.network_previous))
    data={k:np.concatenate(v) for k,v in pieces.items()};n=len(data['a']);device='cuda' if torch.cuda.is_available() else 'cpu';p=p.to(device);t={k:torch.tensor(v,device=device) for k,v in data.items()}
    with torch.no_grad():
        basic=CoordinationPolicy.features(p,t['a'],t['initial'],t['phase'],t['omega'],t['variation'])
        state_x=p.state_features(t['a'],t['initial'],t['phase'],t['omega'],t['variation'],t['previous']);intent_x=p.intent_features(t['a'],t['initial'],t['phase'],t['omega'],t['variation'],t['state'])
        if not args.preserve_normalization:
            p.state_mean.copy_(state_x.mean(0));p.state_scale.copy_(state_x.std(0).clamp_min(.05));p.intent_mean.copy_(intent_x.mean(0));p.intent_scale.copy_(intent_x.std(0).clamp_min(.05))
        memory_count=18 if args.release_memory else 15
        p.state_scale[-memory_count:]=1.;p.intent_scale[36:36+memory_count]=1.
        if args.manifold:target=(p.encode_labels(t['a'],t['initial'],t['phase'],t['target'])-p.baseline_parameters(t['a'],t['initial'],t['phase']))/p.parameter_scale
        else:target=(t['target']-p.base_intent(t['a'],t['initial'],t['phase']))/p.output_scale
        if args.manifold and args.anchor_initial:target[t['state']==0,:16]=0.
        if args.conditional:target[t['state']==0,:30]=0.
        if args.release_memory:
            target[t['state']==0]=0.
            target[(t['state']==15)|(t['state']==16),:30]=0.
    if args.learn_transition_prior:
        counts=torch.bincount(t['previous']*memory_count+t['state'],minlength=memory_count**2).reshape(memory_count,memory_count).float()+1e-5
        p.transition_net.log_transition_prior.copy_(torch.log(counts/counts.sum(-1,keepdim=True)))
    if args.freeze_intent:p.intent_net.requires_grad_(False)
    optimizer=torch.optim.Adam(list(p.intent_net.parameters())+list(p.transition_net.parameters()),lr=args.lr);sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.03)
    weight=torch.tensor(([20.,20.,8.,8.,2.,2.,2.,1.]*2+[2.]*12) if args.manifold else (([4.]*3+[1.]*6+[.2]*3+[.1]*3)*2+[2.]*12),device=device);transition_weight=torch.where(t['state']!=t['previous'],50.,1.);losses=[]
    for epoch in range(args.epochs):
        total=0.;correct=0
        for ids in torch.randperm(n,device=device).split(1024):
            ix=(intent_x[ids]-p.intent_mean)/p.intent_scale;sx=(state_x[ids]-p.state_mean)/p.state_scale
            if args.noise:
                ix=ix+args.noise*torch.randn_like(ix)*torch.cat([torch.ones(36,device=device),torch.zeros(memory_count,device=device),torch.ones(ix.shape[1]-36-memory_count,device=device)])
                sx=sx+args.noise*torch.randn_like(sx)*torch.cat([torch.ones(188,device=device),torch.zeros(memory_count,device=device)])
            pred=p.intent_net(ix);logits=p.transition_net(sx)
            loss_pose=(((pred-target[ids]).square()*weight).mean(-1)*t['sample_weight'][ids]).mean();loss_state=(torch.nn.functional.cross_entropy(logits,t['state'][ids],reduction='none')*transition_weight[ids]*t['sample_weight'][ids]).mean()
            loss=loss_pose+.1*loss_state
            if args.action_weight and not args.manifold:
                intent=p.base_intent(t['a'][ids],t['initial'][ids],t['phase'][ids])+pred*p.output_scale
                predicted=p.action_from_intent(t['a'][ids],intent,t['omega'][ids]).clamp(-1,1)
                action_error=(predicted-t['action'][ids]).square().mean(-1)
                if args.release_memory:action_error=action_error*(t['state'][ids]!=0)
                loss=loss+args.action_weight*(action_error*t['sample_weight'][ids]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(p.intent_net.parameters())+list(p.transition_net.parameters()),5.);optimizer.step();total+=loss.item()*len(ids);correct+=(logits.argmax(-1)==t['state'][ids]).sum().item()
        losses.append(total/n);sched.step()
        if epoch%20==0 or epoch==args.epochs-1:print(json.dumps(dict(epoch=epoch,loss=losses[-1],state_accuracy=correct/n,samples=n,device=device)),flush=True)
    p=p.cpu().eval();(ReleaseMemoryModel(p) if args.release_memory else ConditionalMemoryModel(p) if args.conditional else ManifoldModel(p) if args.manifold else IntentStateModel(p)).save(args.output/'teacher.zip');snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['driving_release_memory.py','driving_conditional_memory.py','driving_manifold_policy.py','driving_intent_state_policy.py','driving_coordination_policy.py','driving_pose_policy.py','driving_task.py','evaluate.py']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    config=dict(policy_variant='release-neural-memory-driving-v1' if args.release_memory else 'conditional-neural-memory-driving-v1' if args.conditional else 'neural-manifold-memory-driving-v1' if args.manifold else 'neural-intent-memory-driving-v1',parent=str(args.model),training_method='BC-intent-and-neural-transition-supervision',stage_labels_at_training=True,initial_geometry_anchor=bool(args.manifold and args.anchor_initial),learned_transition_prior=args.learn_transition_prior,freeze_intent=args.freeze_intent,full_feedback=args.full_feedback,action_loss_weight=args.action_weight,input_noise_std=args.noise,failed_prefixes_used=args.prefixes or bool(args.prefix_data),prefix_data=[str(path) for path in args.prefix_data],corrective_weight=args.corrective_weight,network_previous_used=args.network_previous,preserve_normalization=args.preserve_normalization,safe_complete_labels_used=args.safe_complete,stage_or_expert_at_execution=False,transition_mask=False,actor_dim=152,action_dim=24,critic_updated=False,seed=args.seed,epochs=args.epochs,samples=n,status='requires_physical_development',task=json.loads((args.data[0]/'metadata.json').read_text())['task'],model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest())
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');(args.output/'data_sources.json').write_text(json.dumps(sources,indent=2)+'\n');(args.output/'losses.json').write_text(json.dumps(losses)+'\n');print(json.dumps(config),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--data',type=Path,nargs='+',required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--epochs',type=int,default=500);a.add_argument('--lr',type=float,default=3e-4);a.add_argument('--seed',type=int,default=660);a.add_argument('--prefix-data',type=Path,nargs='*',default=[]);a.add_argument('--corrective-weight',type=float,default=1.);a.add_argument('--network-previous',action='store_true');a.add_argument('--preserve-normalization',action='store_true');a.add_argument('--learn-transition-prior',action='store_true');a.add_argument('--freeze-intent',action='store_true');a.add_argument('--release-memory',action='store_true');a.add_argument('--full-feedback',action='store_true');a.add_argument('--prefixes',action='store_true');a.add_argument('--action-weight',type=float,default=0.);a.add_argument('--conditional',action='store_true');a.add_argument('--safe-complete',action='store_true');a.add_argument('--noise',type=float,default=0.);a.add_argument('--manifold',action='store_true');a.add_argument('--anchor-initial',action='store_true')
    from unified_steering.structured.driving_intent_state_policy import train as canonical_train
    canonical_train(a.parse_args())
