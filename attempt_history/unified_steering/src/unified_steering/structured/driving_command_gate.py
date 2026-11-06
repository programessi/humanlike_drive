"""Fit a bounded neural mixture from legitimate upstream command ranges.

This preserves the qualified small-angle expert and the learned large-angle
basis in one checkpoint and one24-output actor. No hand-stage command enters
at runtime. The interface change is explicit and requires new regression.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from .driving_path_basis import PathBasisModel

def train(model,output):
 torch.set_num_threads(1);torch.manual_seed(711);rng=np.random.default_rng(711);p=PathBasisModel.load(model).policy;output.mkdir(parents=True,exist_ok=False)
 # Only declared task ranges: <=90deg retention and nominal180deg driving.
 amplitudes=np.r_[rng.uniform(0,.5,3000),rng.uniform(.99,1.01,3000)].astype(np.float32)
 modes=rng.integers(0,3,len(amplitudes));ranges=np.c_[np.where(modes!=0,-amplitudes,0),np.where(modes!=1,amplitudes,0)]
 # Reset-angle jitter can be in the supplied sequence even on one-sided tasks.
 ranges[:,0]=np.minimum(ranges[:,0],rng.uniform(-.04,.04,len(ranges)));ranges[:,1]=np.maximum(ranges[:,1],rng.uniform(-.04,.04,len(ranges)))
 x=torch.tensor(ranges,dtype=torch.float32);target=torch.tensor(np.where(amplitudes>.75,2.,-1.),dtype=torch.float32)[:,None];opt=torch.optim.Adam(p.range_gate.parameters(),lr=.01)
 for epoch in range(1200):
  loss=(p.range_gate(x)-target).square().mean();opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad():
  raw=p.range_gate(x);prob=raw.clamp(0,1);expected=(amplitudes>.75).astype(np.float32);error=float(np.max(abs(prob[:,0].numpy()-expected)))
  if error>0:raise RuntimeError('Neural command mixture not exact on declared training ranges')
 p.command_gate_enabled.fill_(True);PathBasisModel(p).save(output/'teacher.zip');meta=json.loads(model.with_name('config.json').read_text());meta.update(parent=str(model),parent_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),command_gate=True,command_gate_network=[2,16,1],command_gate_training_ranges='<=90deg retention, nominal180deg driving; intervening ranges not validated',command_gate_fit_samples=len(x),command_gate_fit_max_error=error,upstream_command_range=True,extra_observation=dict(command_range='2 radian bounds of provided executable angle sequence'),policy_structure='learned bounded neural mixture of frozen qualified R1 and fitted R2 basis; one24-output actor',status='requires_qualified_R1_R2_and_old240_regression',model_sha256=hashlib.sha256((output/'teacher.zip').read_bytes()).hexdigest());(output/'config.json').write_text(json.dumps(meta,indent=2)+'\n');snap=output/'source_snapshot';snap.mkdir()
 for name in ['driving_command_gate.py','driving_path_basis.py','driving_task.py','driving_learning.py','evaluate.py','driving_pose_policy.py','driving_coordination_policy.py']:shutil.copy2(Path(__file__).with_name(name),snap/name)
 print(json.dumps(dict(output=str(output),fit_error=error,loss=float(loss))),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();train(args.model,args.output)
