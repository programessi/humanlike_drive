"""Training-only initialization of a local hand output layer from label data.

The same single actor retains its wrist decoder and all 24 trainable outputs.
No demonstration or prescribed hand target is queried during evaluation.
"""
from pathlib import Path
import json,numpy as np,torch,hashlib
from unified_steering.structured.evaluate import load_teacher
from unified_steering.structured.driving_memory import MemoryModel
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data',type=Path,nargs='+',required=True);args=parser.parse_args()
out=args.output;out.mkdir(exist_ok=False)
m=load_teacher(args.model);p=m.policy
paths=args.data
x=[];y=[]
for path in paths:
 for f in sorted((path/'labels').glob('*.npz')):
  z=np.load(f);a=z['actor'];initial=a[0,14:26]+.15*a[0,66:78]
  x.append(np.c_[a[:,14:26],a[:,66:78],np.tile(initial,(len(a),1))]);y.append(z['action'][:,12:])
x=np.concatenate(x);y=np.concatenate(y)
w=np.zeros((12,36));bias=np.zeros(12)
# Fit desired-grip dependence from training data; explicit local features prevent
# feedback coefficients becoming entangled with the slow changing wrist state.
# The shared negative feedback slopes are also estimated from finger labels.
f=np.c_[x[:,2],x[:,14],x[:,26],np.ones(len(x))];coef=np.linalg.lstsq(f,y[:,2],rcond=None)[0];kq,kg=coef[:2]
for i in range(12):
 target=y[:,i]-kq*x[:,i]-kg*x[:,12+i]
 z=np.c_[x[:,24+i],np.ones(len(x))];co=np.linalg.lstsq(z,target,rcond=None)[0]
 w[i,i]=kq;w[i,12+i]=kg;w[i,24+i]=co[0];bias[i]=co[1]
p.hand_skip=torch.nn.Linear(36,12);p.local_hand=True
with torch.no_grad():
 p.hand_skip.weight.copy_(torch.tensor(w,dtype=torch.float32));p.hand_skip.bias.copy_(torch.tensor(bias,dtype=torch.float32))
 p.head.weight[12:].zero_();p.head.bias[12:].zero_()
p.eval();p.reset();m.save(out/'teacher.zip')
(out/'source_snapshot').mkdir();import shutil;shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
metadata=json.loads(args.model.with_name('config.json').read_text());metadata.update(parent=str(args.model),policy_variant='phase-memory-local-hand-v2',experiment='trainable-local-hand-affine-initialization',fitted_hand_rmse=np.sqrt(np.mean((x@w.T+bias-y)**2,axis=0)).tolist(),all24_mean_action_channels_trainable=True,status='requires_development',shared_slopes=[float(kq),float(kg)],sources=[str(v) for v in paths],model_sha256=hashlib.sha256((out/'teacher.zip').read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
(out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata))
