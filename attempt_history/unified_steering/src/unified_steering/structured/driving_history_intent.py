"""Autoregressive neural intention memory for driving.

Expert action-stage/mode labels are used at training only. Runtime memories
are previous neural intention, predicted categorical state and its duration.
No expert, prescribed transitions, physical-state writes or trajectory reads.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, shutil
import numpy as np
import torch
from torch import nn
from .driving_coordination_policy import CoordinationPolicy, states
from .driving_pose_policy import exp_rotation,rotation6

COUNT=32
STAGES=['clear','relocate','ready','approach','insert','close','unload']

class HistoryHead(nn.Module):
    def __init__(self,transition=False):
        super().__init__();self.transition=transition
        self.networks=nn.ModuleList([nn.Sequential(nn.Linear(231,192),nn.ELU(),nn.Linear(192,192),nn.ELU(),nn.Linear(192,COUNT if transition else 42)) for _ in range(COUNT)])
        self.register_buffer('log_prior',torch.zeros(COUNT,COUNT))
        self.register_buffer('shared_phase',torch.tensor(False))
        if transition:self.phase_net=nn.Sequential(nn.Linear(6,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,COUNT))
        for network in self.networks:
            nn.init.zeros_(network[-1].weight);nn.init.zeros_(network[-1].bias)
    def forward(self,x):
        state=x[:,-COUNT:].argmax(-1);result=x.new_zeros((len(x),COUNT if self.transition else 42))
        for s,net in enumerate(self.networks):
            ids=torch.where(state==s)[0]
            if len(ids):result[ids]=net(x[ids,:231])
        if self.transition:
            result=result+self.log_prior[state]
            if self.shared_phase:
                shared=self.phase_net(x[:,[120,170,183,186,187,128]])
                result=torch.where(((state==0)|(state==15)|(state==16))[:,None],shared,result)
            return result
        return torch.cat([result[:,:30]*((state!=0)&(state!=15)&(state!=16))[:,None],result[:,30:]*(state!=0)[:,None]],-1)

class HistoryIntentPolicy(CoordinationPolicy):
    def __init__(self):
        super().__init__(36);self.intent_net=HistoryHead();self.transition_net=HistoryHead(True)
        self.register_buffer('bounded_intent',torch.tensor(False));self.register_buffer('relative_feedback',torch.tensor(False));self.register_buffer('anchored_hand',torch.tensor(False));self.register_buffer('correct_stage_age',torch.tensor(False));self.register_buffer('feature_mean',torch.zeros(263));self.register_buffer('feature_scale',torch.ones(263));self.reset()
    def reset(self):
        super().reset();self.memory=0;self.memory_age=0.;self.previous_intent=None
    def features_history(self,a,initial,phase,omega,variation,previous,age,state):
        basic=CoordinationPolicy.features(self,a,initial,phase,omega,variation)
        anchor=self.base_intent(a,initial,phase)
        if self.anchored_hand:anchor=torch.cat([anchor[:,:30],initial[:,-12:]],-1)
        prior=(previous-anchor)/self.output_scale
        if self.relative_feedback:
            fk=a[:,102:120].reshape(-1,2,9);gap=a[:,140:152].reshape(-1,2,6)
            position=fk[:,:,:3]*.2+gap[:,:,:3]*.02;rotation=exp_rotation(gap[:,:,3:]*.2)@rotation6(fk[:,:,3:])
            pp=previous[:,:30].reshape(-1,2,15)
            relative=torch.cat([(pp[:,:,:3]-position)/.05,(pp[:,:,3:9]-rotation[:,:,:,:2].reshape(-1,2,6))/.4,pp[:,:,9:12]/.2,pp[:,:,12:15]],-1).reshape(-1,30)
            prior=torch.cat([relative,(previous[:,30:]-initial[:,-12:])/.5],-1)
        return torch.cat([a,basic,prior,age[:,None],torch.nn.functional.one_hot(state,COUNT).to(a.dtype)],-1)
    def reference_intent(self,a,initial,phase,state):
        anchor=self.base_intent(a,initial,phase)
        if self.anchored_hand:anchor=torch.cat([anchor[:,:30],torch.where((state==0)[:,None],anchor[:,30:],initial[:,-12:])],-1)
        return anchor
    def sanitize_intent(self,a,intent,state):
        if not self.bounded_intent:return intent
        clean=torch.nan_to_num(intent,nan=0.,posinf=1e4,neginf=-1e4).clamp(-1e4,1e4)
        pose=clean[:,:30].reshape(-1,2,15)
        axial=pose[:,:,:1].clamp(-.14,.1);yz=pose[:,:,1:3];radius=torch.linalg.vector_norm(yz,dim=-1,keepdim=True)
        fallback_yz=a[:,102:120].reshape(-1,2,9)[:,:,1:3]
        fallback_norm=torch.linalg.vector_norm(fallback_yz,dim=-1,keepdim=True)
        default_yz=torch.zeros_like(yz);default_yz[:,:,0]=1.
        direction=torch.where((radius>1e-6),yz/radius.clamp_min(1e-6),torch.where(fallback_norm>1e-6,fallback_yz/fallback_norm.clamp_min(1e-6),default_yz))
        yz=direction*radius.clamp(.085,.28)
        rotation=rotation6(pose[:,:,3:9]);fallback=rotation6(a[:,102:120].reshape(-1,2,9)[:,:,3:9])
        valid=(torch.linalg.det(rotation)>.9)[:,:,None,None];rotation=torch.where(valid,rotation,fallback)
        safe_pose=torch.cat([axial,yz,rotation[:,:,:,:2].reshape(-1,2,6),pose[:,:,9:12].clamp(-.4,.4),pose[:,:,12:15].clamp(-4.,4.)],-1).reshape(-1,30)
        lo=intent.new_tensor([-.07,.03,.03,.03,.03,.03]*2);hi=intent.new_tensor([1.27,.46,1.67,1.67,1.67,1.67]*2)
        hand=torch.maximum(torch.minimum(clean[:,30:],hi),lo)
        safe=torch.cat([safe_pose,hand],-1)
        return torch.where((state==0)[:,None],intent,safe)
    def predict(self,observation,deterministic=True):
        a=torch.as_tensor(observation['actor'],dtype=torch.float32)
        with torch.no_grad():
            if self.initial is None:
                self.initial=torch.cat([a[102:121],a[14:26]+.15*a[66:78]]).clone()
                self.previous_intent=self.base_intent(a[None],self.initial[None],torch.zeros(1))[0]
            omega=self.base.velocity_head(a[122:129]).reshape(1);old=self.phase;measured=(a[120]-self.initial[18]).item()
            self.phase=float(np.clip(self.phase+.05*omega.item(),measured-np.pi/6,measured+np.pi/6));self.variation+=abs(self.phase-old)
            phase=torch.tensor([self.phase]);variation=torch.tensor([self.variation]);initial=self.initial[None];previous=self.previous_intent[None];age=torch.tensor([self.memory_age]);state=torch.tensor([self.memory])
            x=self.features_history(a[None],initial,phase,omega,variation,previous,age,state)
            new=int(self.transition_net((x-self.feature_mean)/self.feature_scale).argmax(-1).item())
            intent_age=torch.zeros_like(age) if self.correct_stage_age and new!=self.memory else age
            x=self.features_history(a[None],initial,phase,omega,variation,previous,intent_age,torch.tensor([new]))
            intent=self.reference_intent(a[None],initial,phase,torch.tensor([new]))+self.intent_net((x-self.feature_mean)/self.feature_scale)*self.output_scale
            intent=self.sanitize_intent(a[None],intent,torch.tensor([new]))
            self.previous_intent=intent[0].clone();self.memory_age=self.memory_age+.05 if new==self.memory else .05;self.memory=new
            action=self.action_from_intent(a[None],intent,omega)[0]
        return action.clamp(-1,1).numpy(),None

class HistoryIntentModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        p=HistoryIntentPolicy();state=torch.load(path,map_location='cpu',weights_only=True);state.setdefault('correct_stage_age',torch.tensor(False));state.setdefault('anchored_hand',torch.tensor(False));state.setdefault('relative_feedback',torch.tensor(False));state.setdefault('bounded_intent',torch.tensor(False));defaults=p.state_dict()
        for key in defaults:
            if key not in state and ('shared_phase' in key or 'phase_net' in key):state[key]=defaults[key]
        p.load_state_dict(state);p.eval();return cls(p)

def aligned_labels(directory,path):
    trace=np.load(directory/'traces'/path.name);a=np.load(path)['actor'];protocol=json.loads((directory/'protocol.json').read_text());threshold=np.deg2rad(protocol.get('shuffle_parameters',{}).get('release_start_deg',5))
    if 'expert_action_stage' not in trace:raise ValueError('Action-aligned collection required')
    labels=[]
    for i,(stage,mode,side) in enumerate(zip(trace['expert_action_stage'],trace['expert_action_mode'],trace['expert_action_side'])):
        if stage=='initial':labels.append((15+int(a[i,128]>0)) if abs(a[i,120]-a[0,120])>threshold else 0)
        elif stage=='dual':labels.append(17)
        else:labels.append((18 if mode=='restore' else 1)+int(side)*7+STAGES.index(str(stage)))
    state=np.array(labels,np.int64);previous=np.r_[0,state[:-1]];age=np.zeros(len(state),np.float32)
    for i in range(1,len(state)):age[i]=age[i-1]+.05 if previous[i]==previous[i-1] else .05
    return state,previous,age

def actual_memory_supervision(expert_state,expert_previous,network_state):
    """Correct actions in the used memory; supervise transitions only when aligned."""
    network_state=np.asarray(network_state,dtype=np.int64)
    previous=np.r_[0,network_state[:-1]]
    return network_state.copy(),(np.asarray(expert_previous)==previous).astype(np.float32)

def train(args):
    from .evaluate import load_teacher
    torch.set_num_threads(1);torch.manual_seed(args.seed);args.output.mkdir(parents=True,exist_ok=False)
    parent=load_teacher(args.model).policy;p=deepcopy(parent) if isinstance(parent,HistoryIntentPolicy) else HistoryIntentPolicy()
    if args.teacher_contact:
        from .driving_contact_history import ContactHistoryPolicy,ContactHistoryModel
        if not isinstance(p,ContactHistoryPolicy):p=ContactHistoryPolicy(p)
    if not isinstance(parent,HistoryIntentPolicy):p.base=deepcopy(parent.base if hasattr(parent,'base') else parent)
    if args.factor_transition:
        from .driving_factor_transition import FactorTransition,FactorContactModel
        if not isinstance(p.transition_net,FactorTransition):p.transition_net=FactorTransition(p.transition_net.phase_net)
    if args.reset_transition:
        old=p.transition_net
        if args.teacher_contact:
            from .driving_contact_history import ContactHead
            p.transition_net=ContactHead(True)
        else:p.transition_net=HistoryHead(True)
        p.transition_net.phase_net=deepcopy(old.phase_net);p.transition_net.shared_phase.copy_(old.shared_phase)
    p.base.requires_grad_(False);p.hand_gain.requires_grad_(False)
    if args.freeze_intent:p.intent_net.requires_grad_(False)
    if args.correct_stage_age:p.correct_stage_age.fill_(True)
    if args.anchored_hand:p.anchored_hand.fill_(True)
    if args.relative_feedback:p.relative_feedback.fill_(True)
    if args.shared_phase:p.transition_net.shared_phase.fill_(True)
    pieces=[];sources=[]
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            record=json.loads((directory/'cases'/f'{path.stem}.json').read_text())
            corrective=directory in args.corrective_data
            if not record['passed'] and not corrective:continue
            z=np.load(path);n=len(z['actor']);state,previous,age=aligned_labels(directory,path)
            if corrective:
                tr=np.load(directory/'traces'/path.name);errors=abs(tr['error_deg']);bad=np.flatnonzero(np.convolve((errors>args.prefix_error).astype(int),np.ones(5,dtype=int),'valid')>=5)
                n=min(n-20,int(bad[0]) if len(bad) else n)
                if n<40:continue
                previous=np.r_[0,tr['learned_intent_memory'][:-1]].astype(np.int64)
                age=np.r_[0,tr['learned_memory_age'][:-1]].astype(np.float32)
            contact=None
            if args.teacher_contact:
                if 'teacher_contact' in z:contact=z['teacher_contact'][:n].astype(np.float32)
                else:
                    tr=np.load(directory/'traces'/path.name)
                    contact=np.c_[z['privileged'][:n,38:40]*5,np.r_[np.zeros((1,2)),tr['torques_nm'][:n-1]]/.5].astype(np.float32)
            intent_state=state[:n].copy();transition_valid=np.ones(n,np.float32)
            if corrective and args.actual_intent_memory:
                _,expert_previous,_=aligned_labels(directory,path)
                intent_state,transition_valid=actual_memory_supervision(state[:n],expert_previous[:n],tr['learned_intent_memory'][:n])
            a=z['actor'][:n].astype(np.float32);initial,phase,omega,variation=states(a,p);target=np.c_[z['pose'][:n],z['hand_target'][:n]].astype(np.float32)
            with torch.no_grad():start=p.base_intent(torch.tensor(a[:1]),torch.tensor(initial[:1]),torch.zeros(1)).numpy()
            prev=np.r_[start,target[:-1]]
            if corrective:
                prev=np.r_[start,tr['learned_previous_intent'][:n-1]].astype(np.float32)
                if not np.isfinite(prev).all():raise ValueError('Corrective trace lacks actual neural intention history')
            pieces.append(dict(a=a,initial=initial,phase=phase,omega=omega,variation=variation,target=target,previous_intent=prev,state=state[:n],intent_state=intent_state,transition_valid=transition_valid,previous=previous[:n],age=age[:n],action=z['action'][:n],weight=np.full(n,args.corrective_weight if corrective else 1.,np.float32)))
            if contact is not None:pieces[-1]['contact']=contact
            sources.append(dict(path=str(path),samples=n,passed=record['passed'],corrective=corrective,actual_intent_memory=bool(corrective and args.actual_intent_memory),valid_transition_labels=int(transition_valid.sum())))
    device='cuda' if torch.cuda.is_available() else 'cpu';p=p.to(device);t={k:torch.tensor(np.concatenate([piece[k] for piece in pieces]),device=device) for k in pieces[0]};n=len(t['a'])
    with torch.no_grad():
        if args.teacher_contact:p.contact=t['contact']
        x=p.features_history(t['a'],t['initial'],t['phase'],t['omega'],t['variation'],t['previous_intent'],t['age'],t['intent_state']);sx=x.clone();sx[:,-COUNT:]=torch.nn.functional.one_hot(t['previous'],COUNT).float()
        if p.correct_stage_age:x[t['intent_state']!=t['previous'],230]=0.
        if not isinstance(parent,HistoryIntentPolicy) or args.relative_feedback:p.feature_mean.copy_(x.mean(0));p.feature_scale.copy_(x.std(0).clamp_min(.05));p.feature_scale[-COUNT:]=1.
        target=(t['target']-p.reference_intent(t['a'],t['initial'],t['phase'],t['intent_state']))/p.output_scale;target[t['intent_state']==0]=0.;target[(t['intent_state']==15)|(t['intent_state']==16),:30]=0.
        counts=torch.bincount(t['previous']*COUNT+t['state'],minlength=COUNT**2).reshape(COUNT,COUNT).float()+1e-5;p.transition_net.log_prior.copy_((counts/counts.sum(-1,keepdim=True)).log())
        x=(x-p.feature_mean)/p.feature_scale;sx=(sx-p.feature_mean)/p.feature_scale
    parameters=list(p.intent_net.parameters())+list(p.transition_net.parameters());optimizer=torch.optim.Adam(parameters,lr=args.lr);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.03)
    dimension_weight=torch.tensor(([4.]*3+[1.]*6+[.2]*3+[.1]*3)*2+[2.]*12,device=device);losses=[]
    for epoch in range(args.epochs):
        total=0.;correct=0
        for ids in torch.randperm(n,device=device).split(1024):
            xx=x[ids].clone();ss=sx[ids].clone()
            if args.noise:
                noise=args.noise*torch.randn_like(xx[:,:231]);xx[:,:231]+=noise;ss[:,:231]+=noise
            pred=p.intent_net(xx);logits=p.transition_net(ss)
            pose=((pred-target[ids]).square()*dimension_weight).mean(-1)
            event_weight=torch.where(t['state'][ids]!=t['previous'][ids],args.event_weight,1.)
            state_loss=torch.nn.functional.cross_entropy(logits,t['state'][ids],reduction='none')*event_weight
            intent=p.reference_intent(t['a'][ids],t['initial'][ids],t['phase'][ids],t['intent_state'][ids])+pred*p.output_scale
            intent=p.sanitize_intent(t['a'][ids],intent,t['intent_state'][ids])
            action=p.action_from_intent(t['a'][ids],intent,t['omega'][ids]).clamp(-1,1)
            action_loss=(action-t['action'][ids]).square().mean(-1)*(t['intent_state'][ids]!=0)
            if args.factor_transition:state_loss=p.transition_net.supervised_loss(ss,t['state'][ids],args.event_weight)
            state_loss=state_loss*t['transition_valid'][ids]
            loss=(((.1*state_loss) if args.freeze_intent else (pose+.1*state_loss+args.action_weight*action_loss))*t['weight'][ids]).mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,5.);optimizer.step();total+=loss.item()*len(ids);correct+=(logits.argmax(-1)==t['state'][ids]).sum().item()
        scheduler.step();losses.append(total/n)
        if epoch%20==0 or epoch==args.epochs-1:print(json.dumps(dict(epoch=epoch,loss=total/n,state_accuracy=correct/n,samples=n,device=device)),flush=True)
    p=p.cpu().eval();(FactorContactModel(p) if args.factor_transition else ContactHistoryModel(p) if args.teacher_contact else HistoryIntentModel(p)).save(args.output/'teacher.zip')
    config=dict(policy_variant='factorized-contact-history-driving-v1' if args.factor_transition else 'privileged-contact-history-driving-v1' if args.teacher_contact else 'autoregressive-history-intent-driving-v1',parent=str(args.model),training_method='BC-DAgger-action-aligned-intention-history',samples=n,epochs=args.epochs,seed=args.seed,actual_intent_memory=args.actual_intent_memory,freeze_intent=args.freeze_intent,reset_transition=args.reset_transition,actor_dim=152,action_dim=24,privileged_used=bool(args.teacher_contact),teacher_contact_dim=4 if args.teacher_contact else 0,neural_memory_categories=COUNT,previous_neural_intention=True,previous_memory_age=True,correct_stage_age=bool(p.correct_stage_age),anchored_hand=bool(p.anchored_hand),relative_feedback=bool(p.relative_feedback),bounded_intent=bool(p.bounded_intent),shared_phase=bool(p.transition_net.shared_phase),phase_and_restore_labels_at_training_only=True,transition_mask=False,script_at_execution=False,critic_updated=False,task=json.loads((args.data[0]/'metadata.json').read_text())['task'],model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest(),status='requires_physical_development')
    config['task']['teacher_contact_observation']=bool(args.teacher_contact)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');(args.output/'data_sources.json').write_text(json.dumps(sources,indent=2)+'\n');(args.output/'losses.json').write_text(json.dumps(losses)+'\n');snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['driving_factor_transition.py','driving_contact_history.py','driving_history_intent.py','driving_coordination_policy.py','driving_pose_policy.py','driving_learning.py','driving_shuffle_expert.py','driving_task.py','evaluate.py']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    print(json.dumps(config),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--data',type=Path,nargs='+',required=True);a.add_argument('--corrective-data',type=Path,nargs='*',default=[]);a.add_argument('--corrective-weight',type=float,default=3.);a.add_argument('--prefix-error',type=float,default=5.);a.add_argument('--output',type=Path,required=True);a.add_argument('--epochs',type=int,default=400);a.add_argument('--lr',type=float,default=3e-4);a.add_argument('--noise',type=float,default=.003);a.add_argument('--action-weight',type=float,default=2.);a.add_argument('--event-weight',type=float,default=10.);a.add_argument('--seed',type=int,default=677);a.add_argument('--correct-stage-age',action='store_true');a.add_argument('--anchored-hand',action='store_true');a.add_argument('--relative-feedback',action='store_true');a.add_argument('--shared-phase',action='store_true');a.add_argument('--teacher-contact',action='store_true');a.add_argument('--freeze-intent',action='store_true');a.add_argument('--reset-transition',action='store_true');a.add_argument('--factor-transition',action='store_true');a.add_argument('--actual-intent-memory',action='store_true')
    from unified_steering.structured.driving_history_intent import train as canonical_train
    canonical_train(a.parse_args())
