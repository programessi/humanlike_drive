"""Bounded normalized RBF neural memory fitted from supervised intentions.

All centres and values are learned checkpoint parameters. Runtime loads no
trajectory or expert, and its memory is predicted from its own observations.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib,shutil
import numpy as np,torch
from torch import nn
from .driving_history_intent import HistoryIntentPolicy,COUNT,aligned_labels
from .driving_coordination_policy import states

K=512
INDICES=[120,128]+list(range(152,188))+list(range(102,120))+list(range(188,231))+list(range(140,152))+list(range(14,26))+list(range(66,78))
class LocalKernel(nn.Module):
    def __init__(self,transition=False):
        super().__init__();self.transition=transition;dim=len(INDICES);out=COUNT if transition else 42
        self.register_buffer('indices',torch.tensor(INDICES));self.register_buffer('centres',torch.zeros(COUNT,K,dim));self.register_buffer('values',torch.zeros(COUNT,K,out));self.register_buffer('valid',torch.zeros(COUNT,K,dtype=torch.bool));self.register_buffer('mean',torch.zeros(COUNT,dim));self.register_buffer('scale',torch.ones(COUNT,dim));self.register_buffer('bandwidth',torch.full((COUNT,),.3))
    def forward(self,x):
        state=x[:,-COUNT:].argmax(-1);out=x.new_zeros((len(x),COUNT if self.transition else 42))
        for s in range(COUNT):
            ids=torch.where(state==s)[0]
            if len(ids) and self.valid[s].any():
                q=(x[ids][:,self.indices]-self.mean[s])/self.scale[s];c=self.centres[s,self.valid[s]];value=self.values[s,self.valid[s]]
                logits=-torch.cdist(q,c).square()/(2*self.bandwidth[s].square())
                out[ids]=logits.softmax(-1)@value
        if self.transition:return out.clamp_min(1e-7).log()
        return torch.cat([out[:,:30]*((state!=0)&(state!=15)&(state!=16))[:,None],out[:,30:]*(state!=0)[:,None]],-1)

class LocalKernelPolicy(HistoryIntentPolicy):
    def __init__(self,intent_kernel=True):
        super().__init__();self.transition_net=LocalKernel(True)
        if intent_kernel:self.intent_net=LocalKernel()
class LocalKernelModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True);p=LocalKernelPolicy('intent_net.centres' in state);state.setdefault('bounded_intent',torch.tensor(False));p.load_state_dict(state);p.eval();return cls(p)

def train(args):
    from .evaluate import load_teacher
    torch.set_num_threads(1);torch.manual_seed(685);args.output.mkdir(parents=True,exist_ok=False);parent=load_teacher(args.model).policy;p=LocalKernelPolicy(args.intent_kernel)
    for name in ['base','feature_mean','feature_scale','output_scale','wrist_scale','correct_stage_age','anchored_hand','relative_feedback']:
        setattr(p,name,deepcopy(getattr(parent,name)))
    if not args.intent_kernel:p.intent_net=deepcopy(parent.intent_net)
    records=[];sources=[]
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            record=json.loads((directory/'cases'/f'{path.stem}.json').read_text());corrective=directory in args.corrective_data
            if not record['passed'] and not corrective:continue
            z=np.load(path);n=len(z['actor']);state,previous,age=aligned_labels(directory,path);trace=np.load(directory/'traces'/path.name)
            if corrective:
                bad=np.flatnonzero(np.convolve((abs(trace['error_deg'])>3).astype(int),np.ones(5,dtype=int),'valid')>=5);n=min(n-20,int(bad[0]) if len(bad) else n)
                if n<40:continue
                previous=np.r_[0,trace['learned_intent_memory'][:-1]].astype(np.int64);age=np.r_[0,trace['learned_memory_age'][:-1]].astype(np.float32)
            a=z['actor'][:n].astype(np.float32);initial,phase,omega,variation=states(a,p);target=np.c_[z['pose'][:n],z['hand_target'][:n]].astype(np.float32)
            start=p.base_intent(torch.tensor(a[:1]),torch.tensor(initial[:1]),torch.zeros(1)).detach().numpy();prior=np.r_[start,trace['learned_previous_intent'][:n-1] if corrective else target[:-1]].astype(np.float32)
            records.append(dict(a=a,initial=initial,phase=phase,omega=omega,variation=variation,target=target,prior=prior,state=state[:n],previous=previous[:n],age=age[:n]));sources.append(dict(path=str(path),samples=n,corrective=corrective,passed=record['passed']))
    device='cuda' if torch.cuda.is_available() else 'cpu';p=p.to(device);t={k:torch.tensor(np.concatenate([r[k] for r in records]),device=device) for k in records[0]}
    with torch.no_grad():
        sx=p.features_history(t['a'],t['initial'],t['phase'],t['omega'],t['variation'],t['prior'],t['age'],t['previous']);age=t['age'].clone()
        if p.correct_stage_age:age[t['state']!=t['previous']]=0.
        x=p.features_history(t['a'],t['initial'],t['phase'],t['omega'],t['variation'],t['prior'],age,t['state']);sx=(sx-p.feature_mean)/p.feature_scale;x=(x-p.feature_mean)/p.feature_scale
        target=(t['target']-p.reference_intent(t['a'],t['initial'],t['phase'],t['state']))/p.output_scale;target[t['state']==0]=0.;target[(t['state']==15)|(t['state']==16),:30]=0.
        for head,xx,group,yy in [(p.transition_net,sx,t['previous'],torch.nn.functional.one_hot(t['state'],COUNT).float()*.9999+.0001/COUNT)]+([(p.intent_net,x,t['state'],target)] if args.intent_kernel else []):
            for s in range(COUNT):
                ids=torch.where(group==s)[0]
                if not len(ids):continue
                q=xx[ids][:,head.indices];head.mean[s]=q.mean(0);head.scale[s]=q.std(0).clamp_min(.3);q=(q-head.mean[s])/head.scale[s]
                events=torch.where(t['state'][ids]!=t['previous'][ids])[0];selected=events[:K//2] if head.transition else torch.empty(0,dtype=torch.long,device=device)
                remaining=K-len(selected);uniform=torch.linspace(0,len(ids)-1,min(remaining,len(ids)),device=device).long();chosen=torch.unique(torch.cat([selected,uniform]),sorted=True)[:K]
                head.centres[s,:len(chosen)]=q[chosen];head.values[s,:len(chosen)]=yy[ids[chosen]];head.valid[s,:len(chosen)]=True;head.bandwidth[s]=args.bandwidth
    p=p.cpu().eval();LocalKernelModel(p).save(args.output/'teacher.zip')
    config=dict(policy_variant='local-normalized-kernel-driving-v1',parent=str(args.model),training_method='supervised-RBF-centre-and-output-fitting',samples=len(t['a']),intent_kernel=args.intent_kernel,normalized_positive_kernel=True,bandwidth=args.bandwidth,script_at_execution=False,expert_or_stage_input_at_execution=False,privileged_used=False,actor_dim=152,action_dim=24,task=json.loads((args.data[0]/'metadata.json').read_text())['task'],status='requires_physical_development',model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest())
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');(args.output/'data_sources.json').write_text(json.dumps(sources,indent=2)+'\n');snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['driving_local_kernel.py','driving_history_intent.py','driving_coordination_policy.py','driving_pose_policy.py','driving_learning.py','evaluate.py']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    print(json.dumps(config),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--data',type=Path,nargs='+',required=True);a.add_argument('--corrective-data',type=Path,nargs='*',default=[]);a.add_argument('--intent-kernel',action='store_true');a.add_argument('--bandwidth',type=float,default=.3);a.add_argument('--output',type=Path,required=True)
    from unified_steering.structured.driving_local_kernel import train as canonical_train
    canonical_train(a.parse_args())
