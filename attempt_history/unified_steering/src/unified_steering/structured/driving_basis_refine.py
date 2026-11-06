"""Bounded direct policy search over already fitted neural basis outputs.

Updates checkpoint weights only. Runtime has no added controller, expert,
stage, simulator writes or reward-dependent action override.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,shutil
import numpy as np,torch
from .driving_path_basis import PathBasisModel

def refine(model,output,radial=0.,axial=0.):
 torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=False);p=PathBasisModel.load(model).policy
 with torch.no_grad():
  pose=p.values[:,:30].reshape(-1,2,15);phase=p.centre_phase[:,None,None];blend=(1-phase.cos())*.5;slope=phase.sin()*.5
  hand=p.values[:,30:].reshape(-1,2,6);closed=((hand[:,:,2:].mean(-1,keepdim=True)-.2)/.8).clamp(0,1)
  yz=pose[:,:,1:3];radius=torch.linalg.vector_norm(yz,dim=-1,keepdim=True).clamp_min(.085);unit=yz/radius;dp=pose[:,:,10:12];du=(dp-unit*(unit*dp).sum(-1,keepdim=True))/radius
  pose[:,:,1:3]+=float(radial)*blend*closed*unit;pose[:,:,10:12]+=float(radial)*closed*(slope*unit+blend*du)
  pose[:,:,:1]+=float(axial)*blend*closed;pose[:,:,9:10]+=float(axial)*slope*closed
 PathBasisModel(p).save(output/'teacher.zip');config=json.loads(model.with_name('config.json').read_text());config.update(parent=str(model),parent_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),training_method='supervised-RBF-plus-bounded-direct-policy-search',weight_search_parameters=dict(radial_m=radial,axial_m=axial),status='requires_physical_development',model_sha256=hashlib.sha256((output/'teacher.zip').read_bytes()).hexdigest());(output/'config.json').write_text(json.dumps(config,indent=2)+'\n');snap=output/'source_snapshot';snap.mkdir()
 for name in ['driving_basis_refine.py','driving_path_basis.py','driving_coordination_policy.py','driving_pose_policy.py','evaluate.py']:shutil.copy2(Path(__file__).with_name(name),snap/name)
 print(json.dumps(dict(output=str(output),parameters=config['weight_search_parameters'])),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--radial',type=float,default=0.);a.add_argument('--axial',type=float,default=0.);x=a.parse_args();refine(x.model,x.output,x.radial,x.axial)
