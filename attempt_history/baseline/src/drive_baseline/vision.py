"""ResNet-18 waypoint predictor; compatible with official ImageNet backbone weights."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .navigation import turn_command

LOOKAHEAD_METRES = (3., 6., 9., 12., 15.)


class BasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(incoming,outgoing,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(outgoing)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(outgoing,outgoing,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(outgoing)
        self.downsample=nn.Sequential(nn.Conv2d(incoming,outgoing,1,stride,bias=False),nn.BatchNorm2d(outgoing)) if incoming!=outgoing or stride!=1 else None

    def forward(self,x):
        skip=x if self.downsample is None else self.downsample(x)
        out=self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out))+skip)


class ResNet18(nn.Module):
    def __init__(self,pool_size=(1,1)):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,2,1)
        self.layer1=nn.Sequential(BasicBlock(64,64),BasicBlock(64,64))
        self.layer2=nn.Sequential(BasicBlock(64,128,2),BasicBlock(128,128))
        self.layer3=nn.Sequential(BasicBlock(128,256,2),BasicBlock(256,256))
        self.layer4=nn.Sequential(BasicBlock(256,512,2),BasicBlock(512,512))
        self.avgpool=nn.AdaptiveAvgPool2d(pool_size)

    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


class WaypointPolicy(nn.Module):
    def __init__(self,spatial_pool=(1,1),predict_xy=False,navigation_branches=False):
        super().__init__()
        self.spatial_pool=tuple(spatial_pool)
        self.predict_xy=predict_xy
        self.navigation_branches=navigation_branches
        self.backbone=ResNet18(self.spatial_pool)
        features=512*self.spatial_pool[0]*self.spatial_pool[1]+4
        self.head=nn.Sequential(nn.Linear(features,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10 if predict_xy else 5))
        if predict_xy:
            with torch.no_grad():
                self.head[4].weight[::2].zero_()
                self.head[4].bias[::2].copy_(torch.tensor(LOOKAHEAD_METRES))
        if navigation_branches:
            self.branches=nn.ModuleList([nn.Sequential(copy.deepcopy(self.head[2]),nn.ReLU(),copy.deepcopy(self.head[4])) for _ in range(3)])
            self.head[2].requires_grad_(False);self.head[4].requires_grad_(False)

    def load_imagenet(self,path):
        state=torch.load(path,map_location="cpu",weights_only=True)
        state={k:v for k,v in state.items() if not k.startswith("fc.")}
        self.backbone.load_state_dict(state,strict=True)

    def forward(self,rgb,speed,command):
        features=self.backbone(rgb)
        fused=torch.cat([features,speed.reshape(-1,1)/10,command],dim=1)
        if self.navigation_branches:
            shared=self.head[1](self.head[0](fused))
            outputs=torch.stack([branch(shared) for branch in self.branches],dim=1)
            output=outputs[torch.arange(len(rgb),device=rgb.device),command.argmax(dim=1)]
        else:
            output=self.head(fused)
        return output.reshape(-1,5,2) if self.predict_xy else output

    def warm_start(self,payload):
        """Keep a trained visual backbone while upgrading spatial/path heads."""
        state=payload["state_dict"]
        if state["head.0.weight"].shape==self.head[0].weight.shape and state["head.4.weight"].shape==self.head[4].weight.shape:
            if self.navigation_branches and not any(k.startswith("branches.") for k in state):
                missing=self.load_state_dict(state,strict=False)
                if missing.unexpected_keys or any(not k.startswith("branches.") for k in missing.missing_keys):
                    raise ValueError(f"Unexpected warm-start keys: {missing}")
                self._initialize_branches()
            else:
                self.load_state_dict(state)
            return
        self.backbone.load_state_dict({k[len("backbone."):]:v for k,v in state.items() if k.startswith("backbone.")})
        old_first=state["head.0.weight"]
        if old_first.shape[1]!=516:
            raise ValueError("Warm-start shape conversion requires an original globally pooled model")
        with torch.no_grad():
            blocks=self.spatial_pool[0]*self.spatial_pool[1]
            # Flattened feature order is channel, row, column.
            self.head[0].weight[:,:512*blocks].copy_(old_first[:,:512].repeat_interleave(blocks,dim=1)/blocks)
            self.head[0].weight[:,-4:].copy_(old_first[:,-4:])
            self.head[0].bias.copy_(state["head.0.bias"])
            self.head[2].weight.copy_(state["head.2.weight"])
            self.head[2].bias.copy_(state["head.2.bias"])
            if self.predict_xy and state["head.4.weight"].shape[0]==5:
                self.head[4].weight.zero_();self.head[4].bias.zero_()
                self.head[4].weight[1::2].copy_(state["head.4.weight"])
                self.head[4].bias[1::2].copy_(state["head.4.bias"])
                self.head[4].bias[::2].copy_(torch.tensor(LOOKAHEAD_METRES,device=self.head[4].bias.device))
            else:
                self.head[4].weight.copy_(state["head.4.weight"])
                self.head[4].bias.copy_(state["head.4.bias"])
        if self.navigation_branches:self._initialize_branches()

    def _initialize_branches(self):
        for branch in self.branches:
            branch[0].load_state_dict(self.head[2].state_dict())
            branch[2].load_state_dict(self.head[4].state_dict())


def policy_from_checkpoint(payload):
    model=WaypointPolicy(**payload.get("model_config",{}))
    model.load_state_dict(payload["state_dict"])
    return model


def image_tensor(rgb):
    im=Image.fromarray(np.asarray(rgb,dtype=np.uint8)).resize((384,224))
    tensor=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
    return (tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


class DrivingDataset(Dataset):
    def __init__(self,manifest,split="train",predict_xy=False,command_turns=False,validation_phase=0):
        manifest=Path(manifest)
        self.root=manifest.parent
        self.predict_xy=predict_xy
        records=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        # Temporal blocks prevent adjacent frames leaking across the split.
        self.records=[r for r in records if ((int(r["frame"])//200)%5==validation_phase)==(split=="val")]
        if not self.records:
            raise ValueError(f"No {split} records: collect enough frames for temporal train/val blocks")
        if predict_xy and any("path_waypoints" not in r for r in self.records):
            raise ValueError("Full XY training requires path_waypoints labels; recollect with the current CARLA bridge")
        if command_turns:
            for r in self.records:
                x,y=r["path_waypoints"][-1]
                r["command"]=turn_command(x,y)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        with Image.open(self.root/r["image"]) as im:rgb=np.asarray(im.convert("RGB"))
        command=torch.zeros(3);command[int(r.get("command",0))]=1
        label=r["path_waypoints"] if self.predict_xy else r["lateral_waypoints"]
        return image_tensor(rgb),torch.tensor(r["speed"],dtype=torch.float32),command,torch.tensor(label,dtype=torch.float32)


def train_vision(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model_config={"spatial_pool":args.spatial_pool,"predict_xy":args.predict_xy,"navigation_branches":args.navigation_branches}
    model=WaypointPolicy(**model_config).to(args.device)
    if args.resume:
        model.warm_start(torch.load(args.resume,map_location=args.device,weights_only=True))
    elif args.imagenet:
        model.load_imagenet(args.imagenet)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    train_dataset=DrivingDataset(args.manifest,"train",args.predict_xy,args.command_turns,args.validation_phase)
    sampler=None
    if args.balanced_turns:
        weights=[1+2*np.clip(max(abs(y) for y in r["lateral_waypoints"])/3,0,3) for r in train_dataset.records]
        sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
    train=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=0)
    val=DataLoader(DrivingDataset(args.manifest,"val",args.predict_xy,args.command_turns,args.validation_phase),batch_size=args.batch_size,shuffle=False,num_workers=0)
    args.output.mkdir(parents=True,exist_ok=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(train_samples=len(train.dataset),val_samples=len(val.dataset),
                  validation_split=f"(frame // 200) % 5 == {args.validation_phase}; same-map temporal blocks",
                  train_commands=[sum(r.get("command",0)==i for r in train.dataset.records) for i in range(3)],
                  val_commands=[sum(r.get("command",0)==i for r in val.dataset.records) for i in range(3)],
                  optimizer="AdamW",weight_decay=1e-4,loss="SmoothL1",image_size=[224,384])
    (args.output/"config.json").write_text(json.dumps(config,indent=2))
    best=float("inf")
    history=[]
    for epoch in range(args.epochs):
        metrics={}
        for phase,loader in (("train",train),("val",val)):
            model.train(phase=="train")
            total,absolute,lateral_absolute,count=0.,0.,0.,0
            with torch.set_grad_enabled(phase=="train"):
                for rgb,speed,command,ys in loader:
                    rgb,speed,command,ys=[x.to(args.device) for x in (rgb,speed,command,ys)]
                    pred=model(rgb,speed,command)
                    loss=nn.functional.smooth_l1_loss(pred,ys)
                    if phase=="train":
                        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    total+=float(loss.detach())*len(rgb);count+=len(rgb)
                    absolute+=float((pred-ys).abs().mean().detach())*len(rgb)
                    lateral_absolute+=float(((pred[...,1]-ys[...,1]) if args.predict_xy else pred-ys).abs().mean().detach())*len(rgb)
            metrics[phase]=total/count
            metrics[f"{phase}_mae_m"]=absolute/count
            metrics[f"{phase}_lateral_mae_m"]=lateral_absolute/count
        history.append(dict(epoch=epoch+1,**metrics))
        print(history[-1],flush=True)
        payload={"state_dict":model.state_dict(),"model_config":model_config,"lookahead":LOOKAHEAD_METRES,"history":history,"seed":args.seed,"imagenet":str(args.imagenet) if args.imagenet else None,"resume":str(args.resume) if args.resume else None}
        torch.save(payload,args.output/"last.pt")
        if metrics["val"]<best:
            best=metrics["val"];torch.save(payload,args.output/"best.pt")
        (args.output/"history.json").write_text(json.dumps(history,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("runs/vision"))
    p.add_argument("--imagenet",type=Path)
    p.add_argument("--resume",type=Path)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--spatial-pool",type=int,nargs=2,default=[1,1])
    p.add_argument("--predict-xy",action="store_true")
    p.add_argument("--balanced-turns",action="store_true")
    p.add_argument("--navigation-branches",action="store_true")
    p.add_argument("--command-turns",action="store_true",help="Derive coarse training commands from the 15m teacher endpoint")
    p.add_argument("--validation-phase",type=int,choices=range(5),default=0)
    train_vision(p.parse_args())
