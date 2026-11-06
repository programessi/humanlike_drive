"""Conservative PPO refinement with actor-preserving critic adaptation and labels.

Experts produce training targets only. Saved candidates are ordinary standalone
24-action teachers evaluated by driving_learning without this training wrapper.
"""
from pathlib import Path
from typing import NamedTuple
import argparse,json,hashlib
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import SubprocVecEnv
from .driving_task import DrivingEnv,DrivingConfig
from .legacy.regrasp.controller import RegraspControlConfig
from stable_baselines3.common.monitor import Monitor
from .demonstrator import DemonstratorConfig
from .driving_roll_integral_audit import TangentRollDemonstrator
from . import driving_roll_integral_audit as roll
from .evaluate import load_teacher
from .driving_learning import write

class LabelSamples(NamedTuple):
    observations:dict;actions:torch.Tensor;old_values:torch.Tensor;old_log_prob:torch.Tensor
    advantages:torch.Tensor;returns:torch.Tensor;expert_actions:torch.Tensor

class LabelBuffer(DictRolloutBuffer):
    def reset(self):
        super().reset();self.labels=np.zeros((self.buffer_size,self.n_envs,24),np.float32)
    def get(self,batch_size=None):
        if not self.generator_ready:self.labels=self.swap_and_flatten(self.labels)
        yield from super().get(batch_size)
    def _get_samples(self,ids,env=None):
        return LabelSamples(*super()._get_samples(ids,env),self.to_torch(self.labels[ids]))

class TrainingEnv(DrivingEnv):
    def __init__(self,protocol,rank):
        self.protocol,self.rank,self.episode=protocol,rank,0
        task=dict(protocol['task']);task['controller']=RegraspControlConfig(**task['controller'])
        super().__init__(DrivingConfig(**task))
    def reset(self,seed=None,options=None):
        self.episode+=1
        if options is None:
            cases=self.protocol['training_cases'];case=cases[(self.episode+self.rank)%len(cases)]
            amplitude,direction,template=case
            self.config.amplitude_deg=amplitude
            options=dict(free_side=-1,direction=direction,template=template,exact_curriculum=True)
            seed=self.protocol['training_seed_start']+self.rank*10000+self.episode
        obs,info=super().reset(seed=seed,options=options)
        self.expert=TangentRollDemonstrator(self,DemonstratorConfig(**self.protocol['demonstrator']))
        return obs,info
    def step(self,action):
        label=self.expert.predict(self._observation())[0]
        obs,reward,done,truncated,info=super().step(action)
        info['training_expert_action']=label
        return obs,reward,done,truncated,info

class LabelCallback(BaseCallback):
    def __init__(self,output,save_steps):super().__init__();self.output=output;self.save_steps=save_steps;self.last_save=0
    def _on_step(self):
        self.model.rollout_buffer.labels[self.model.rollout_buffer.pos]=np.array([i['training_expert_action'] for i in self.locals['infos']])
        if self.num_timesteps-self.last_save>=self.save_steps:
            save_candidate(self.model,self.output/f'step_{self.num_timesteps:07d}.zip');self.last_save=self.num_timesteps
        return True

def save_candidate(model,path):
    optimizer=model.policy.optimizer
    model.policy.optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-5,eps=1e-5)
    try:model.save(path,exclude=['retention'])
    finally:model.policy.optimizer=optimizer

class LabelPPO(PPO):
    def train(self):
        self.policy.set_training_mode(True);pgs=[];vls=[];bcs=[];kls=[];clips=[]
        for epoch in range(self.n_epochs):
            stop=False
            for data in self.rollout_buffer.get(self.batch_size):
                dist=self.policy.get_distribution(data.observations)
                log_prob=dist.log_prob(data.actions)
                features=self.policy.extract_features(data.observations)
                values=self.policy.value_net(self.policy.mlp_extractor.forward_critic(features.detach())).flatten()
                advantages=data.advantages
                if len(advantages)>1:advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio=torch.exp(log_prob-data.old_log_prob)
                pg=-torch.min(advantages*ratio,advantages*torch.clamp(ratio,.95,1.05)).mean()
                vf=(data.returns-values).square().mean()
                weights=torch.tensor([4.]*12+[1.]*12)
                bc=((dist.distribution.mean-data.expert_actions).square()*weights).mean()
                ids=torch.randint(len(self.retention['actor']),(256,))
                ob={k:self.retention[k][ids] for k in ['actor','privileged']}
                keep=self.policy.get_distribution(ob).distribution.mean
                retention=(keep-self.retention['action'][ids]).square().mean()
                loss=pg+.5*vf+self.bc_coef*bc+5.*retention
                with torch.no_grad():
                    lr=log_prob-data.old_log_prob;kl=torch.mean((torch.exp(lr)-1)-lr).item()
                if kl>.015:stop=True;break
                self.policy.optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),.5);self.policy.optimizer.step()
                pgs.append(pg.item());vls.append(vf.item());bcs.append(bc.item());kls.append(kl)
                clips.append(torch.mean((abs(ratio-1)>.05).float()).item())
            if stop:break
        self._n_updates+=epoch+1
        for name,values in [('policy_loss',pgs),('value_loss',vls),('expert_bc_loss',bcs),('approx_kl',kls),('clip_fraction',clips)]:
            self.logger.record('train/'+name,np.mean(values) if values else 0.)
        self.logger.record('train/explained_variance',explained_variance(self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')


def adapt_critic(model,directories,output,epochs=40):
    arrays={k:[] for k in ['actor','privileged','returns']}
    for directory in directories:
        for path in sorted((directory/'labels').glob('*.npz')):
            trace=np.load(directory/'traces'/path.name);labels=np.load(path)
            rewards=trace['reward'];returns=np.zeros(len(rewards),np.float32);value=0.
            for t in range(len(rewards)-1,-1,-1):value=float(rewards[t])+.995*value;returns[t]=value
            for key in ['actor','privileged']:arrays[key].append(labels[key])
            arrays['returns'].append(returns)
    x={k:torch.from_numpy(np.concatenate(v)) for k,v in arrays.items()}
    obs={key:x[key][:512] for key in ['actor','privileged']}
    with torch.no_grad():before=model.policy.get_distribution(obs).distribution.mean.clone()
    params=list(model.policy.mlp_extractor.critic.parameters())+list(model.policy.value_net.parameters())
    opt=torch.optim.Adam(params,lr=3e-4);losses=[]
    for epoch in range(epochs):
        total=0.
        for ids in torch.randperm(len(x['returns'])).split(1024):
            with torch.no_grad():f=model.policy.extract_features({k:x[k][ids] for k in ['actor','privileged']})
            v=model.policy.value_net(model.policy.mlp_extractor.forward_critic(f)).flatten()
            loss=(v-x['returns'][ids]).square().mean();opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(ids)
        losses.append(total/len(x['returns']))
    with torch.no_grad():after=model.policy.get_distribution(obs).distribution.mean
    assert torch.equal(before,after),'Critic adaptation changed actor'
    write(output/'critic_adaptation.json',dict(losses=losses,actor_bitwise_unchanged=True,samples=len(x['returns']),targets='finite actual rollout discounted returns; no expert action in evaluation'))


def main(args):
    protocol=json.loads(args.protocol.read_text());torch.set_num_threads(1);torch.manual_seed(protocol['seed'])
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'protocol.json',protocol)
    write(args.output/'config.json',dict(interface=protocol['interface'],task=protocol['task'],training_method='PPO-plus-BC-DAgger',
        parent=str(args.model),parent_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),status='training',script_at_evaluation=False))
    roll._TABLE=protocol['tangent_geometry'];roll.TangentRollDemonstrator.phase_lead_deg=protocol['phase_lead_deg']
    model=load_teacher(args.model);adapt_critic(model,args.data,args.output,protocol.get('critic_epochs',40));save_candidate(model,args.output/'critic_adapted.zip')
    env=SubprocVecEnv([lambda rank=rank:Monitor(TrainingEnv(protocol,rank)) for rank in range(4)],start_method='fork')
    model.__class__=LabelPPO;model.set_env(env);model.n_steps=256;model.batch_size=256;model.n_epochs=3
    model.rollout_buffer=LabelBuffer(256,model.observation_space,model.action_space,device='cpu',gamma=.995,gae_lambda=.95,n_envs=4)
    model.bc_coef=protocol.get('bc_coef',20.)
    z=np.load(protocol['retention_dataset']);model.retention={k:torch.from_numpy(z[k]) for k in ['actor','privileged','action']}
    actor=list(model.policy.features_extractor.parameters())+list(model.policy.mlp_extractor.actor.parameters())+list(model.policy.action_net.parameters())
    critic=list(model.policy.mlp_extractor.critic.parameters())+list(model.policy.value_net.parameters())
    with torch.no_grad():model.policy.log_std[:12]=np.log(protocol.get('wrist_std',.01));model.policy.log_std[12:]=np.log(.005)
    model.policy.log_std.requires_grad_(False)
    model.policy.optimizer=torch.optim.Adam([dict(params=actor,lr=protocol.get('actor_lr',3e-6)),dict(params=critic,lr=3e-4)],eps=1e-5)
    model.tensorboard_log=str(args.output/'tensorboard');model.verbose=1
    model.learn(total_timesteps=protocol['steps'],callback=LabelCallback(args.output,16384),reset_num_timesteps=True,tb_log_name='PPO')
    save_candidate(model,args.output/'final.zip');env.close()
    metadata=json.loads((args.output/'config.json').read_text());metadata.update(status='trained_requires_development',actual_steps=model.num_timesteps,critic_updated=True)
    write(args.output/'config.json',metadata)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,nargs='+',required=True)
    main(p.parse_args())
