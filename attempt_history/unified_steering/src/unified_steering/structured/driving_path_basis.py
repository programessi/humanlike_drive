"""Supervised normalized radial-basis actor with autonomous phase memory.

The checkpoint contains learned centres/outputs, not executable expert logic.
Inference reads only current observations and its integrated command history.
No external stage, reference-template ID, planned elapsed time or expert runs.
This is an imitation baseline and is not PPO or a deployable student.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib,shutil
import numpy as np, torch
from torch import nn
from .driving_coordination_policy import CoordinationPolicy,states
from .driving_pose_policy import rotation6,exp_rotation,log_rotation

def transport_pose_jet(pose,delta):
 """Integrate the local spatial twist, preserving rigid circular motion."""
 w=pose[...,12:15]*delta;rotation=exp_rotation(w);angle=torch.linalg.vector_norm(w,dim=-1,keepdim=True)
 x,y,z=w.unbind(-1);zero=torch.zeros_like(x);skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*w.shape[:-1],3,3)
 eye=torch.eye(3,dtype=w.dtype,device=w.device).expand_as(skew);square=angle.square()
 second=.5*torch.sinc(angle/(2*torch.pi)).square()
 third=torch.where(square<1e-4,1/6-square/120+square.square()/5040,(1-torch.sinc(angle/torch.pi))/square.clamp_min(1e-12))
 jacobian=eye+second[...,None]*skew+third[...,None]*(skew@skew)
 pos=pose[...,:3]+torch.einsum('...ij,...j->...i',jacobian,pose[...,9:12]*delta)
 rot=rotation@rotation6(pose[...,3:9]);dp=torch.einsum('...ij,...j->...i',rotation,pose[...,9:12])
 return torch.cat([pos,rot[...,:,:2].reshape(*pose.shape[:-1],6),dp,pose[...,12:15]],-1)

class PathBasisPolicy(CoordinationPolicy):
 def __init__(self,count=1,sensory=False,hand_sensory=False):
  super().__init__();self.register_buffer('phase_bound',torch.tensor(np.pi/6,dtype=torch.float64));self.register_buffer('renew_hand_anchor',torch.tensor(True));self.anchor_gate=nn.Sequential(nn.Linear(7,32),nn.ELU(),nn.Linear(32,32),nn.ELU(),nn.Linear(32,2));self.register_buffer('anchor_gate_enabled',torch.tensor(False));self.range_gate=nn.Sequential(nn.Linear(2,16),nn.ELU(),nn.Linear(16,1));self.register_buffer('axis_frame',torch.tensor(False));self.register_buffer('phase_leak',torch.tensor(0.));self.register_buffer('rigid_jet',torch.tensor(False));self.register_buffer('command_gate_enabled',torch.tensor(False));self.register_buffer('centre_fk',torch.zeros(count,18));self.register_buffer('centre_wheel',torch.zeros(count));self.register_buffer('anchor_gain',torch.zeros(count,2));self.register_buffer('centre_phase',torch.zeros(count));self.register_buffer('command_query',torch.tensor(False));self.register_buffer('use_jets',torch.tensor(False));self.register_buffer('hand_sensory',torch.tensor(hand_sensory));self.register_buffer('log_weight',torch.zeros(count));self.register_buffer('sensory',torch.tensor(sensory));self.register_buffer('centres',torch.zeros(count,(20 if sensory else 2)+(12 if hand_sensory else 0)));self.register_buffer('values',torch.zeros(count,42));self.register_buffer('metric',torch.tensor([1.,.04]+(([.5]*3+[.08]*6)*2 if sensory else [])+([.02]*12 if hand_sensory else [])));self.register_buffer('bandwidth',torch.tensor(.004));self.register_buffer('canonical_initial',torch.zeros(31));self.reset()
 def reset(self):
  super().reset();self.reference_previous=None;self.reference_variation=0.;self.previous_intent=None;self.internal_rebase_count=0
 def predict(self,observation,deterministic=True):
  a=torch.as_tensor(observation['actor'],dtype=torch.float32)
  if self.command_gate_enabled:
   if 'command_range' not in observation:raise ValueError('Versioned learned command gate requires upstream command_range in radians')
   limits=torch.as_tensor(observation['command_range'],dtype=a.dtype)
   if limits.shape!=(2,) or not torch.isfinite(limits).all() or limits[0]>limits[1]:raise ValueError('Expected finite ordered command_range')
  with torch.no_grad():
   if self.initial is None:self.initial=torch.cat([a[102:121],a[14:26]+.15*a[66:78]]).clone()
   omega=self.base.velocity_head(a[122:129]).reshape(1);before=self.phase;measured=(a[120]-self.initial[18]).item()
   route=self.range_gate(limits/np.pi).clamp(0.,1.) if self.command_gate_enabled else omega.new_ones(1)
   omega=omega-route*self.phase_leak*(self.phase-measured)
   limit=np.pi/6+route.item()*(float(self.phase_bound)-np.pi/6);self.phase=float(np.clip(self.phase+.05*omega.item(),measured-limit,measured+limit));self.variation+=abs(self.phase-before)
   reference=float((a[120]+a[122]*np.deg2rad(10.)-self.initial[18]).item())
   if self.reference_previous is not None:self.reference_variation+=abs(reference-self.reference_previous)
   self.reference_previous=reference
   if self.anchor_gate_enabled:
    command=a[120]+a[122]*np.deg2rad(10.)
    gate_input=torch.stack([command/np.pi,a.new_tensor(self.reference_variation/np.pi),a[128],a[16:20].mean(),a[22:26].mean(),a[120]/np.pi,a[122]])
    scores=self.anchor_gate(gate_input);scores=torch.stack([scores[0],scores[1]*route[0]+(1-route[0])*(scores[0]-1.)])
    if int(scores.argmax())==1:
     hand_anchor=self.initial[-12:].clone();self.initial=torch.cat([a[102:121],a[14:26]+.15*a[66:78]]).clone()
     if not self.renew_hand_anchor:self.initial[-12:]=hand_anchor
     self.phase=0.;self.variation=0.;self.reference_variation=0.;self.internal_rebase_count+=1
     reference=float((command-self.initial[18]).item());self.reference_previous=reference
   query=a.new_tensor([reference,self.reference_variation]) if self.command_query else a.new_tensor([self.phase,self.variation])
   if self.sensory:
    current=a[102:120].reshape(2,9);ini=self.initial[:18].reshape(2,9);canonical=self.canonical_initial[:18].reshape(2,9);inv=rotation6(canonical[:,3:])@rotation6(ini[:,3:]).transpose(-1,-2)
    pos=canonical[:,:3]+torch.einsum('hij,hj->hi',inv,current[:,:3]-ini[:,:3]);rot=inv@rotation6(current[:,3:]);sensor=torch.cat([pos,rot[:,:,:2].reshape(2,6)],-1)
    if self.axis_frame:
     from .driving_frame import canonical_wrist_state
     sensor=canonical_wrist_state(a,self.initial,self.canonical_initial)
    query=torch.cat([query,sensor.reshape(-1)])
   if self.hand_sensory:query=torch.cat([query,a[14:26]-self.initial[-12:]+self.canonical_initial[-12:]])
   distance=((self.centres-query)*self.metric).square().sum(-1)
   ids=distance.topk(min(16,len(distance)),largest=False).indices
   weight=(-distance[ids]/(2*self.bandwidth.square())+self.log_weight[ids]).softmax(0)
   candidate=self.values[ids]
   if self.use_jets:
    pose=candidate[:,:30].reshape(-1,2,15);delta=(self.phase-self.centre_phase[ids]).clamp(-.52,.52)[:,None,None]
    pos=pose[:,:,:3]+pose[:,:,9:12]*delta;rot=exp_rotation(pose[:,:,12:15]*delta)@rotation6(pose[:,:,3:9])
    transported=transport_pose_jet(pose,delta) if self.rigid_jet else torch.cat([pos,rot[:,:,:,:2].reshape(-1,2,6),pose[:,:,9:]],-1)
    candidate=torch.cat([transported.reshape(-1,30),candidate[:,30:]],-1)
   if self.anchor_gain.any():
    current=a[102:120].reshape(2,9);ini=self.initial[:18].reshape(2,9);canonical=self.canonical_initial[:18].reshape(2,9);inv=rotation6(canonical[:,3:])@rotation6(ini[:,3:]).transpose(-1,-2)
    measured_position=(canonical[:,:3]+torch.einsum('hij,hj->hi',inv,current[:,:3]-ini[:,:3]))*.2;measured_rotation=inv@rotation6(current[:,3:])
    template=self.centre_fk[ids].reshape(-1,2,9);pose=candidate[:,:30].reshape(-1,2,15);delta=(measured-self.centre_wheel[ids])[:,None,None]
    expected_position=template[:,:,:3]*.2+pose[:,:,9:12]*delta;expected_rotation=exp_rotation(pose[:,:,12:15]*delta)@rotation6(template[:,:,3:9]);gain=self.anchor_gain[ids,:,None]
    pos=pose[:,:,:3]+gain*(measured_position-expected_position);correction=log_rotation(measured_rotation@expected_rotation.transpose(-1,-2));rot=exp_rotation(gain*correction)@rotation6(pose[:,:,3:9])
    candidate=torch.cat([torch.cat([pos,rot[:,:,:,:2].reshape(-1,2,6),pose[:,:,9:]],-1).reshape(-1,30),candidate[:,30:]],-1)
   intent=(weight[:,None]*candidate).sum(0)
   canonical=self.canonical_initial[:18].reshape(2,9);actual=self.initial[:18].reshape(2,9)
   # Per-reset rigid frame adaptation; all motion is still predicted by the RBF.
   transform=rotation6(actual[:,3:])@rotation6(canonical[:,3:]).transpose(-1,-2)
   pose=intent[:30].reshape(2,15)
   pos=actual[:,:3]*.2+torch.einsum('hij,hj->hi',transform,pose[:,:3]-canonical[:,:3]*.2)
   rot=transform@rotation6(pose[:,3:9]);dp=torch.einsum('hij,hj->hi',transform,pose[:,9:12]);dr=torch.einsum('hij,hj->hi',transform,pose[:,12:15])
   hand=intent[30:]+self.initial[-12:]-self.canonical_initial[-12:]
   adapted=torch.cat([pos,rot[:,:,:2].reshape(2,6),dp,dr],-1)
   if self.axis_frame:
    from .driving_frame import adapt_pose
    adapted=adapt_pose(pose,self.initial,self.canonical_initial)
   intent=torch.cat([adapted.reshape(30),hand])
   self.previous_intent=intent.clone();action=self.action_from_intent(a[None],intent[None],omega)[0]
   if self.command_gate_enabled:
    blend=self.range_gate(limits/np.pi).clamp(0.,1.)
    basic=self.base.action(a[None],self.initial[None],a.new_tensor([self.phase]),omega)[0]
    units=a.new_tensor(([.05/self.wrist_scale[0]]*3+[np.deg2rad(40.)/self.wrist_scale[1]]*3)*2+[1.]*12)
    action=blend*action+(1-blend)*basic*units
  return action.clamp(-1,1).numpy(),None
class PathBasisModel:
 def __init__(self,p):self.policy=p
 def reset(self):self.policy.reset()
 def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
 def save(self,path):torch.save(self.policy.state_dict(),path)
 @classmethod
 def load(cls,path):
  state=torch.load(path,map_location='cpu',weights_only=True);p=PathBasisPolicy(len(state['centres']),bool(state.get('sensory',torch.tensor(False))),bool(state.get('hand_sensory',torch.tensor(False))));state.setdefault('hand_sensory',torch.tensor(False));state.setdefault('log_weight',torch.zeros(len(state['centres'])));state.setdefault('centre_fk',torch.zeros(len(state['centres']),18));state.setdefault('centre_wheel',torch.zeros(len(state['centres'])));state.setdefault('anchor_gain',torch.zeros(len(state['centres']),2));state.setdefault('sensory',torch.tensor(False));state.setdefault('centre_phase',torch.zeros(len(state['centres'])));state.setdefault('command_query',torch.tensor(False));state.setdefault('use_jets',torch.tensor(False));defaults=p.state_dict()
  for name in defaults:
   if name not in state and (name.startswith(('range_gate.','anchor_gate.')) or name in ('anchor_gate_enabled','command_gate_enabled','rigid_jet','phase_leak','axis_frame','renew_hand_anchor','phase_bound')):state[name]=defaults[name]
  p.load_state_dict(state);p.eval();return cls(p)

def train(args):
 from .evaluate import load_teacher
 torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False);parent=load_teacher(args.model).policy
 proxy=CoordinationPolicy();proxy.base=deepcopy(parent.base if hasattr(parent,'base') else parent)
 arrays=[];source=[];canonical=None
 for directory in args.data:
  for path in sorted((directory/'labels').glob('*.npz')):
   case=json.loads((directory/'cases'/f'{path.stem}.json').read_text())
   corrective=directory in args.corrective_data
   if (not case['passed'] and not corrective) or (not corrective and args.seed>=0 and (case['template']!='sequence' or case['seed']!=args.seed)):continue
   z=np.load(path);trace=np.load(directory/'traces'/path.name);n=len(z['actor'])
   if corrective:
    bad=np.flatnonzero(np.convolve((abs(trace['error_deg'])>3).astype(int),np.ones(5,dtype=int),'valid')>=5);n=min(n-20,int(bad[0]) if len(bad) else n)
    # A shadow expert stuck restoring at the next roundtrip is not a valid oracle.
    legs=case.get('timeline',{}).get('segments',[])
    rebases=[event for event in case.get('expert_events',[]) or [] if event.get('event')=='collection_expert_memory_rebased']
    if len(legs)>2 and not rebases:n=min(n,round(legs[2]['start']/.05))
    for event in case.get('expert_events',[]) or []:
     if event.get('position_offset_mm',0)>20 or event.get('orientation_offset_deg',0)>15:n=min(n,round(event['time']/.05))
    if n<40:continue
   a=z['actor'][:n].astype(np.float32);initial,phase,omega,variation=states(a,proxy);target=np.c_[z['pose'][:n],z['hand_target'][:n]].astype(np.float32)
   if canonical is None:canonical=initial[0].copy()
   actual=torch.tensor(initial[0,:18].reshape(2,9));base=torch.tensor(canonical[:18].reshape(2,9));transform=rotation6(base[:,3:])@rotation6(actual[:,3:]).transpose(-1,-2)
   pose=torch.tensor(target[:,:30].reshape(-1,2,15));pos=base[:,:3]*.2+torch.einsum('hij,nhj->nhi',transform,pose[:,:,:3]-actual[:,:3]*.2);rot=transform@rotation6(pose[:,:,3:9]);dp=torch.einsum('hij,nhj->nhi',transform,pose[:,:,9:12]);dr=torch.einsum('hij,nhj->nhi',transform,pose[:,:,12:15]);hand=torch.tensor(target[:,30:]-initial[:,-12:]+canonical[-12:]);target=torch.cat([torch.cat([pos,rot[:,:,:,:2].reshape(-1,2,6),dp,dr],-1).reshape(-1,30),hand],-1).numpy()
   # Remove near-duplicate holding samples but retain every moving sample.
   reference=a[:,120]+a[:,122]*np.deg2rad(10.)-initial[0,18];refvariation=np.r_[0.,np.cumsum(abs(np.diff(reference)))]
   query=np.c_[reference,refvariation] if args.command_query else np.c_[phase,variation]
   if args.sensory or args.anchor_gain:
    current=torch.tensor(a[:,102:120].reshape(-1,2,9));pos=base[:,:3]+torch.einsum('hij,nhj->nhi',transform,current[:,:,:3]-actual[:,:3]);rot=transform@rotation6(current[:,:,3:]);sensor=torch.cat([pos,rot[:,:,:,:2].reshape(-1,2,6)],-1).reshape(-1,18).numpy()
    if args.sensory:query=np.c_[query,sensor]
   else:sensor=a[:,102:120]
   if args.hand_sensory:query=np.c_[query,a[:,14:26]-initial[:,-12:]+canonical[-12:]]
   select=np.r_[np.linalg.norm(np.diff(query,axis=0),axis=1)>1e-5,True]
   gain=np.ones((len(a),2),np.float32)
   trace=np.load(directory/'traces'/path.name)
   free=~np.isin(trace['expert_action_stage'][:n],['initial','dual','close','unload']);gain[np.flatnonzero(free),trace['expert_action_side'][:n][free].astype(int)]=0.
   arrays.append((query[select],target[select],phase[select],sensor[select],(a[:,120]-initial[0,18])[select],gain[select]*args.anchor_gain,np.full(select.sum(),np.log(args.corrective_weight) if corrective else 0.,np.float32)));source.append(dict(path=str(path),samples=int(select.sum()),corrective=corrective,passed=case['passed']))
 if not arrays:raise ValueError('No passing sequence source for requested training seed')
 p=PathBasisPolicy(sum(len(x[0]) for x in arrays),args.sensory,args.hand_sensory);p.base=deepcopy(parent.base if hasattr(parent,'base') else parent);p.wrist_scale.copy_(torch.tensor([.1,np.deg2rad(60.)]));p.centres.copy_(torch.tensor(np.concatenate([x[0] for x in arrays])));p.values.copy_(torch.tensor(np.concatenate([x[1] for x in arrays])));p.log_weight.copy_(torch.tensor(np.concatenate([x[6] for x in arrays])));p.centre_fk.copy_(torch.tensor(np.concatenate([x[3] for x in arrays])));p.centre_wheel.copy_(torch.tensor(np.concatenate([x[4] for x in arrays])));p.anchor_gain.copy_(torch.tensor(np.concatenate([x[5] for x in arrays])));p.centre_phase.copy_(torch.tensor(np.concatenate([x[2] for x in arrays])));p.command_query.fill_(args.command_query);p.use_jets.fill_(args.jets);p.canonical_initial.copy_(torch.tensor(canonical));p.bandwidth.fill_(args.bandwidth);p.metric[1]=args.variation_weight
 p.eval();PathBasisModel(p).save(args.output/'teacher.zip');config=dict(policy_variant='normalized-path-basis-driving-v1',parent=str(args.model),training_method='supervised-normalized-RBF-output-fitting',samples=len(p.centres),sources=source,actor_dim=152,action_dim=24,privileged_used=False,script_at_execution=False,stage_at_execution=False,planned_time_or_template_at_execution=False,bandwidth=args.bandwidth,variation_weight=args.variation_weight,command_query=args.command_query,first_order_jet=args.jets,sensory_basis=args.sensory,hand_sensory=args.hand_sensory,corrective_weight=args.corrective_weight,learned_contact_anchor_gain=args.anchor_gain,grasp_role_labels_at_training_only=True,status='requires_physical_development',task=json.loads((args.data[0]/'metadata.json').read_text())['task'],model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest());(args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');snapshot=args.output/'source_snapshot';snapshot.mkdir()
 for name in ['driving_path_basis.py','driving_coordination_policy.py','driving_pose_policy.py','evaluate.py','driving_task.py']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
 print(json.dumps(config),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--data',type=Path,nargs='+',required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--seed',type=int,default=48321);a.add_argument('--bandwidth',type=float,default=.004);a.add_argument('--variation-weight',type=float,default=.04);a.add_argument('--command-query',action='store_true');a.add_argument('--jets',action='store_true');a.add_argument('--sensory',action='store_true');a.add_argument('--anchor-gain',type=float,default=0.);a.add_argument('--hand-sensory',action='store_true');a.add_argument('--corrective-data',type=Path,nargs='*',default=[]);a.add_argument('--corrective-weight',type=float,default=10.)
 from unified_steering.structured.driving_path_basis import train as canonical_train
 canonical_train(a.parse_args())
