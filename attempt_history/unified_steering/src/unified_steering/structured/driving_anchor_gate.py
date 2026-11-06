"""Learn when the neural movement anchor can be renewed after a roundtrip.

Only the actor's latent anchor/history are renewed. Robot state, contacts,
reference clock and actuator targets are never reset. Labels exist at training
only; inference uses its own7-input neural memory gate.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from .driving_path_basis import PathBasisModel

def train(model,data,output):
 torch.set_num_threads(1);torch.manual_seed(717);output.mkdir(parents=True,exist_ok=False);p=PathBasisModel.load(model).policy;pieces=[];sources=[]
 for directory in data:
  for path in sorted((directory/'labels').glob('*.npz')):
   record=json.loads((directory/'cases'/f'{path.stem}.json').read_text())
   if not record['passed']:continue
   z=np.load(path);a=z['actor'];tr=np.load(directory/'traces'/path.name);command=a[:,120]+a[:,122]*np.deg2rad(10.);progress=np.r_[0,np.cumsum(abs(np.diff(command)))];label=np.zeros(len(a),np.int64);times=np.arange(len(a))*.05
   segments=record['timeline']['segments']
   for leg in range(0,len(segments),2):
    start=segments[leg]['start'];end=segments[leg+1]['hold_end'];ids=(times>=start)&(times<end)
    origin=progress[max(0,round(start/.05)-1)];progress[ids]-=origin
    stop=segments[leg+1];positive=(times>=stop['end'])&(times<stop['hold_end'])&(tr['expert_action_stage']=='dual');label[positive]=1
   x=np.c_[command/np.pi,progress/np.pi,a[:,128],a[:,16:20].mean(-1),a[:,22:26].mean(-1),a[:,120]/np.pi,a[:,122]].astype(np.float32);pieces.append((x,label));sources.append(str(path))
 x=np.concatenate([r[0] for r in pieces]);y=np.concatenate([r[1] for r in pieces]);rng=np.random.default_rng(717);positive=np.flatnonzero(y);negative=np.flatnonzero(y==0);selected=np.r_[positive,rng.choice(negative,min(len(negative),len(positive)*5),replace=False)];device='cuda' if torch.cuda.is_available() else 'cpu';net=p.anchor_gate.to(device);xx=torch.tensor(x[selected],device=device);yy=torch.tensor(y[selected],device=device);opt=torch.optim.Adam(net.parameters(),lr=.003)
 for epoch in range(500):
  total=0.
  for ids in torch.randperm(len(xx),device=device).split(2048):
   logits=net(xx[ids]);loss=torch.nn.functional.cross_entropy(logits,yy[ids]);opt.zero_grad();loss.backward();opt.step();total+=float(loss)*len(ids)
  if epoch%100==0:print(json.dumps(dict(epoch=epoch,loss=total/len(xx),samples=len(xx))),flush=True)
 p.anchor_gate=net.cpu();p.anchor_gate_enabled.fill_(True);p.eval();PathBasisModel(p).save(output/'teacher.zip');meta=json.loads(model.with_name('config.json').read_text());meta.update(parent=str(model),training_method='supervised-RBF-plus-neural-anchor-renewal',anchor_gate_network=[7,32,32,2],anchor_gate_fit_samples=len(xx),anchor_gate_sources=sources,physical_or_clock_reset=False,internal_anchor_renewal='only predicted neural memory; new wrist/hand anchor from own measured state',status='requires_physical_development',model_sha256=hashlib.sha256((output/'teacher.zip').read_bytes()).hexdigest());(output/'config.json').write_text(json.dumps(meta,indent=2)+'\n');snap=output/'source_snapshot';snap.mkdir()
 for name in ['driving_anchor_gate.py','driving_path_basis.py','driving_task.py','driving_learning.py','evaluate.py','driving_pose_policy.py','driving_coordination_policy.py']:shutil.copy2(Path(__file__).with_name(name),snap/name)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--data',type=Path,nargs='+',required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();train(args.model,args.data,args.output)
