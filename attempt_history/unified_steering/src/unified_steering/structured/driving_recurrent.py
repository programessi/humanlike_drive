"""Learned recurrent residual actor; no execution-time expert or action override."""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
from torch import nn

class RecurrentPolicy(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.features_extractor=deepcopy(base.features_extractor)
        self.actor=deepcopy(base.mlp_extractor.actor)
        self.head=deepcopy(base.action_net)
        self.context=nn.Sequential(nn.Linear(168,128),nn.ELU())
        self.gru=nn.GRU(128,128,batch_first=True)
        self.residual=nn.Linear(128,24)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        self.hidden=None

    def forward(self,actor,privileged,hidden=None):
        batch,length,_=actor.shape
        f=self.features_extractor(dict(actor=actor.reshape(-1,152),privileged=privileged.reshape(-1,54)))[:,:168]
        baseline=self.head(self.actor(f)).reshape(batch,length,24)
        context=self.context(f).reshape(batch,length,128)
        y,hidden=self.gru(context,hidden)
        return baseline+.1*self.residual(y),hidden

    def reset(self):self.hidden=None

    def predict(self,observation,deterministic=True):
        with torch.no_grad():
            actor=torch.as_tensor(observation['actor'],dtype=torch.float32).reshape(1,1,152)
            privileged=torch.as_tensor(observation['privileged'],dtype=torch.float32).reshape(1,1,54)
            y,self.hidden=self(actor,privileged,self.hidden)
        return y[0,0].clamp(-1,1).numpy(),None

class RecurrentModel:
    def __init__(self,policy):self.policy=policy
    def reset(self):self.policy.reset()
    def predict(self,*args,**kwargs):return self.policy.predict(*args,**kwargs)
    def save(self,path):
        from .driving_policy import DrivingFeatures
        torch.save(dict(format='recurrent-driving-v2',state_dict=self.policy.state_dict(),
            normalized=isinstance(self.policy.features_extractor,DrivingFeatures)),path)
    @classmethod
    def load(cls,path):
        import gymnasium as gym
        import pickle,types
        class CompatUnpickler(pickle.Unpickler):
            def find_class(self,module,name):
                if module=='__main__' and name=='RecurrentPolicy':return RecurrentPolicy
                return super().find_class(module,name)
        compat=types.SimpleNamespace(__name__='driving_compat_pickle',Unpickler=CompatUnpickler,load=pickle.load,loads=pickle.loads)
        value=torch.load(path,map_location='cpu',weights_only=False,pickle_module=compat)
        if value.get('format')=='recurrent-driving-v2':
            from .legacy.regrasp.policy import TeacherPolicy
            from .driving_policy import DrivingTeacherPolicy
            spaces=gym.spaces.Dict(actor=gym.spaces.Box(-np.inf,np.inf,(152,),np.float32),
                privileged=gym.spaces.Box(-np.inf,np.inf,(54,),np.float32))
            base=(DrivingTeacherPolicy if value['normalized'] else TeacherPolicy)(spaces,gym.spaces.Box(-1.,1.,(24,),np.float32),lambda _:1e-5)
            policy=RecurrentPolicy(base);policy.load_state_dict(value['state_dict'])
        else:policy=value['policy']
        policy.eval();policy.reset();return cls(policy)



def train(args):
    from .evaluate import load_teacher
    from .driving_learning import write
    protocol=json.loads(args.protocol.read_text());torch.set_num_threads(1)
    torch.manual_seed(protocol['seed']);rng=np.random.default_rng(protocol['seed'])
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'protocol.json',protocol)
    parent=load_teacher(args.model)
    model=parent if isinstance(parent,RecurrentModel) else RecurrentModel(RecurrentPolicy(parent.policy))
    metadata=dict(policy_variant='recurrent-driving-v1',training_method='recurrent-BC-DAgger',critic_updated=False,
        interface=protocol['interface'],task=protocol['task'],parent=str(args.model),
        parent_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),script_at_evaluation=False,
        all24_mean_action_channels_trainable=True,status='training')
    write(args.output/'config.json',metadata)
    episodes=[];sources=[]
    for directory in args.data:
        for path in sorted((directory/'labels').glob('*.npz')):
            z=np.load(path);episodes.append({k:torch.from_numpy(z[k]) for k in ['actor','privileged','action']})
            sources.append(dict(path=str(path),samples=len(z['actor'])))
    write(args.output/'data_sources.json',sources)
    z=np.load(protocol['retention_dataset']);retention={k:torch.from_numpy(z[k]) for k in ['actor','privileged','action']}
    policy=model.policy;policy.train();base=list(policy.features_extractor.parameters())+list(policy.actor.parameters())+list(policy.head.parameters())
    context=list(policy.context.parameters())+list(policy.gru.parameters())+list(policy.residual.parameters())
    optimizer=torch.optim.Adam([dict(params=base,lr=protocol.get('base_lr',1e-5)),dict(params=context,lr=protocol.get('context_lr',1e-4))])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs'],eta_min=3e-7)
    weights=torch.tensor([4.]*12+[1.]*12);losses=[]
    for epoch in range(protocol['epochs']):
        total=count=0
        for order in torch.randperm(len(episodes)).split(protocol.get('episode_batch',6)):
            selected=[episodes[i] for i in order];length=max(len(x['actor']) for x in selected);batch=len(selected)
            arrays={k:torch.zeros(batch,length,v) for k,v in [('actor',152),('privileged',54),('action',24)]}
            valid=torch.zeros(batch,length)
            for i,episode in enumerate(selected):
                n=len(episode['actor']);valid[i,:n]=1
                for key in arrays:arrays[key][i,:n]=episode[key]
            hidden=None
            for start in range(0,length,protocol.get('bptt',128)):
                end=min(length,start+protocol.get('bptt',128));mask=valid[:,start:end]
                mean,hidden=policy(arrays['actor'][:,start:end],arrays['privileged'][:,start:end],hidden)
                err=((mean-arrays['action'][:,start:end]).square()*weights).mean(-1)
                hold=torch.where(arrays['actor'][:,start:end,128].abs()<1e-5,2.,1.)
                loss=(err*mask*hold).sum()/mask.sum().clamp_min(1.)
                ids=torch.from_numpy(rng.integers(0,len(retention['actor']),256))
                keep,_=policy(retention['actor'][ids,None],retention['privileged'][ids,None])
                objective=loss+protocol.get('retention_weight',.2)*(keep[:,0]-retention['action'][ids]).square().mean()
                optimizer.zero_grad();objective.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
                hidden=hidden.detach();total+=loss.item()*mask.sum().item();count+=mask.sum().item()
        scheduler.step();losses.append(total/max(count,1))
        if epoch%5==0 or epoch==protocol['epochs']-1:print(json.dumps(dict(epoch=epoch,loss=losses[-1],samples=count)),flush=True)
    policy.eval();policy.reset();model.save(args.output/'teacher.zip')
    metadata.update(status='trained_requires_development',final_loss=losses[-1],parameters=sum(p.numel() for p in policy.parameters()),
        model_sha256=hashlib.sha256((args.output/'teacher.zip').read_bytes()).hexdigest())
    write(args.output/'config.json',metadata);write(args.output/'losses.json',losses)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,nargs='+',required=True)
    train(p.parse_args())
