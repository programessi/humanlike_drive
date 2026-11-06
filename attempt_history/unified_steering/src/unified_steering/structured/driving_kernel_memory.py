"""Conditional radial-basis neural intentions, fitted by supervised ridge regression.

Basis centres and output weights are checkpoint parameters. Inference has no
expert, trajectory file, stage command, contact truth or geometry table.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib,shutil
import numpy as np,torch
from torch import nn
from .driving_conditional_memory import ConditionalMemoryPolicy
from .driving_intent_state_policy import stage_labels
from .driving_coordination_policy import CoordinationPolicy,states

class KernelIntent(nn.Module):
    count=384
    def __init__(self):
        super().__init__()
        self.register_buffer('indices',torch.tensor([31,32,33,34,35]+list(range(51,69))))
        self.register_buffer('centres',torch.zeros(15,self.count,23))
        self.register_buffer('mean',torch.zeros(15,23));self.register_buffer('scale',torch.ones(15,23))
        self.register_buffer('bandwidth',torch.ones(15));self.register_buffer('weights',torch.zeros(15,self.count+24,42))
    def basis(self,x,s):
        q=(x[:,self.indices]-self.mean[s])/self.scale[s]
        kernel=torch.exp(-torch.cdist(q,self.centres[s]).square()/(2*self.bandwidth[s].square()))
        return torch.cat([torch.ones_like(q[:,:1]),q,kernel],-1)
    def forward(self,x):
        state=x[:,36:51].argmax(-1);result=x.new_zeros((len(x),42))
        for s in range(15):
            ids=torch.where(state==s)[0]
            if len(ids):result[ids]=self.basis(x[ids],s)@self.weights[s]
        return torch.cat([result[:,:30]*(state!=0).to(x.dtype)[:,None],result[:,30:]],-1)

class KernelMemoryPolicy(ConditionalMemoryPolicy):
    def __init__(self):super().__init__();self.intent_net=KernelIntent()
class KernelMemoryModel:
    def __init__(self,p):self.policy=p
    def reset(self):self.policy.reset()
    def predict(self,*a,**kw):return self.policy.predict(*a,**kw)
    def save(self,path):torch.save(self.policy.state_dict(),path)
    @classmethod
    def load(cls,path):
        p=KernelMemoryPolicy();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));p.eval();return cls(p)

def train(args):
    from .evaluate import load_teacher
    torch.set_num_threads(1);torch.manual_seed(671);args.output.mkdir(parents=True,exist_ok=False)
    parent=load_teacher(args.model).policy;p=KernelMemoryPolicy();p.base=deepcopy(parent.base);p.transition_net=deepcopy(parent.transition_net);p.state_mean.copy_(parent.state_mean);p.state_scale.copy_(parent.state_scale)
    pieces=[];sources=[]
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            record=json.loads((directory/'cases'/f'{path.stem}.json').read_text())
            if not record['passed']:continue
            z=np.load(path);a=z['actor'].astype(np.float32);initial,phase,omega,variation=states(a,p);state,_=stage_labels(directory,path,len(a));pieces.append((a,initial,phase,omega,variation,state,np.c_[z['pose'],z['hand_target']].astype(np.float32)));sources.append(str(path))
    values=[np.concatenate([piece[j] for piece in pieces]) for j in range(7)]
    device='cuda' if torch.cuda.is_available() else 'cpu';p=p.to(device);a,initial,phase,omega,variation,state,target=[torch.tensor(v,device=device) for v in values]
    with torch.no_grad():
        basic=CoordinationPolicy.features(p,a,initial,phase,omega,variation);x=torch.cat([basic,torch.nn.functional.one_hot(state,15).float(),a[:,102:120]],-1)
        p.intent_mean.copy_(x.mean(0));p.intent_scale.copy_(x.std(0).clamp_min(.05));p.intent_scale[36:51]=1.;x=(x-p.intent_mean)/p.intent_scale
        y=(target-p.base_intent(a,initial,phase))/p.output_scale;y[state==0,:30]=0.;stats=[]
        for s in range(15):
            ids=torch.where(state==s)[0]
            if not len(ids):continue
            xx=x[ids];yy=y[ids];q=xx[:,p.intent_net.indices];p.intent_net.mean[s]=q.mean(0);p.intent_net.scale[s]=q.std(0).clamp_min(.3)
            q=(q-p.intent_net.mean[s])/p.intent_net.scale[s]
            train_ids=torch.where(torch.arange(len(ids),device=device)%5!=0)[0];valid_ids=torch.where(torch.arange(len(ids),device=device)%5==0)[0]
            selected=train_ids[torch.linspace(0,len(train_ids)-1,p.intent_net.count,device=device).long()];p.intent_net.centres[s]=q[selected]
            best=None
            for bandwidth in [.3,.6,1.2,2.4]:
                p.intent_net.bandwidth[s]=bandwidth;b=p.intent_net.basis(xx,s).double();bt=b[train_ids];yt=yy[train_ids].double()
                reg=torch.eye(bt.shape[1],device=device,dtype=torch.float64)*1e-6;reg[0,0]=1e-10
                weights=torch.linalg.solve(bt.T@bt+reg,bt.T@yt)
                mse=(b[valid_ids]@weights-yy[valid_ids]).square().mean().item()
                if best is None or mse<best[0]:best=(mse,bandwidth,weights)
            p.intent_net.bandwidth[s]=best[1];p.intent_net.weights[s]=best[2].float();stats.append(dict(state=s,samples=len(ids),interleaved_validation_mse=best[0],bandwidth=best[1]));print(json.dumps(stats[-1]),flush=True)
    p=p.cpu().eval();KernelMemoryModel(p).save(args.output/'teacher.zip')
    config=dict(policy_variant='kernel-neural-memory-driving-v1',parent=str(args.model),training_method='supervised-radial-basis-ridge-intentions-with-learned-conditional-memory',runtime_expert=False,runtime_stage_input=False,runtime_trajectory_file=False,critic_updated=False,samples=len(a),task=json.loads((args.data[0]/'metadata.json').read_text())['task'],model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest(),status='requires_physical_development',fit_stats=stats)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');(args.output/'data_sources.json').write_text(json.dumps(sources,indent=2)+'\n');snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['driving_kernel_memory.py','driving_conditional_memory.py','driving_intent_state_policy.py','driving_coordination_policy.py','driving_pose_policy.py','evaluate.py']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--data',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    from unified_steering.structured.driving_kernel_memory import train as canonical_train
    canonical_train(parser.parse_args())
