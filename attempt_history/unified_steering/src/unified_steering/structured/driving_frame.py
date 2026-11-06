"""Wheel-axis equivariant coordinates for renewing learned grasp anchors.

A changed initial palm orientation must not tilt the wheel's rotation axis.
Positions adapt in cylindrical wheel coordinates; palm orientation offsets
act on the right of the learned relative orientation.
"""
import torch
from .driving_pose_policy import rotation6,exp_rotation

def _parameters(initial,canonical):
 actual=initial[:18].reshape(2,9);source=canonical[:18].reshape(2,9)
 beta=torch.atan2(actual[:,2],actual[:,1])-torch.atan2(source[:,2],source[:,1])
 vector=torch.stack([beta,torch.zeros_like(beta),torch.zeros_like(beta)],-1)
 turn=exp_rotation(vector);radial=torch.linalg.vector_norm(actual[:,1:3],dim=-1)-torch.linalg.vector_norm(source[:,1:3],dim=-1)
 axial=(actual[:,0]-source[:,0])*.2
 return actual,source,turn,radial*.2,axial

def canonical_wrist_state(actor,initial,canonical):
 actual,source,turn,radial,axial=_parameters(initial,canonical);current=actor[102:120].reshape(2,9)
 pos=torch.einsum('hji,hj->hi',turn,current[:,:3]*.2);radius=torch.linalg.vector_norm(pos[:,1:],dim=-1,keepdim=True).clamp_min(.02)
 pos=torch.cat([pos[:,:1]-axial[:,None],pos[:,1:]*(radius-radial[:,None])/radius],-1)
 rotation=turn.transpose(-1,-2)@rotation6(current[:,3:])@rotation6(actual[:,3:]).transpose(-1,-2)@turn@rotation6(source[:,3:])
 return torch.cat([pos/.2,rotation[:,:,:2].reshape(2,6)],-1)

def adapt_pose(pose,initial,canonical):
 actual,source,turn,radial,axial=_parameters(initial,canonical);p=pose[:,:3];radius=torch.linalg.vector_norm(p[:,1:],dim=-1,keepdim=True).clamp_min(.02);unit=p[:,1:]/radius
 local=torch.cat([p[:,:1]+axial[:,None],p[:,1:]*(radius+radial[:,None])/radius],-1);pos=torch.einsum('hij,hj->hi',turn,local)
 rotation=turn@rotation6(pose[:,3:9])@rotation6(source[:,3:]).transpose(-1,-2)@turn.transpose(-1,-2)@rotation6(actual[:,3:])
 tangent=pose[:,10:12]-unit*(unit*pose[:,10:12]).sum(-1,keepdim=True)
 velocity=torch.cat([pose[:,9:10],pose[:,10:12]+radial[:,None]/radius*tangent],-1);dp=torch.einsum('hij,hj->hi',turn,velocity);dr=torch.einsum('hij,hj->hi',turn,pose[:,12:15])
 return torch.cat([pos,rotation[:,:,:2].reshape(2,6),dp,dr],-1)
