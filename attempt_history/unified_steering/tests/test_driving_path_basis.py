import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from unified_steering.structured.driving_path_basis import PathBasisPolicy,PathBasisModel,transport_pose_jet
from unified_steering.structured.evaluate import load_teacher

class PathBasisChecks(unittest.TestCase):
 def make_policy(self):
  p=PathBasisPolicy(3);p.base.velocity_head.weight.data.zero_();p.base.velocity_head.bias.data.zero_()
  a=np.zeros(152,np.float32);identity=np.array([1,0,0,1,0,0],np.float32)
  fk=np.array([[0,.8,0,*identity],[0,-.8,0,*identity]],np.float32);a[102:120]=fk.ravel();a[14:26]=.2
  p.canonical_initial.copy_(torch.tensor(np.r_[a[102:121],a[14:26]]))
  pose=np.array([[0,.16,0,*identity,0,0,0,0,0,0],[0,-.16,0,*identity,0,0,0,0,0,0]],np.float32)
  p.values.copy_(torch.tensor(np.tile(np.r_[pose.ravel(),a[14:26]],(3,1))))
  p.centres.copy_(torch.tensor([[0,0],[1,1],[-1,1]],dtype=torch.float32));return p,a
 def test_reference_progress_is_independent_of_tracking_correction(self):
  p,a=self.make_policy();p.command_query.fill_(True);p.use_jets.fill_(True)
  p.predict({'actor':a});a[122]=.1;p.predict({'actor':a});before=p.reference_variation
  # Same external command with a different measured angle must not advance history.
  a[120]+=.01;a[122]-=.01/np.deg2rad(10.)
  p.predict({'actor':a});self.assertAlmostEqual(before,p.reference_variation,places=6)
 def test_checkpoint_reload_and_reset_reproduce_closed_policy_memory(self):
  p,a=self.make_policy();p.command_query.fill_(True);p.use_jets.fill_(True)
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'teacher.zip';PathBasisModel(p).save(path);q=PathBasisModel.load(path).policy
   for angle in [0.,.02,.04,.01,0.]:
    a[122]=angle/np.deg2rad(10.)
    np.testing.assert_array_equal(p.predict({'actor':a})[0],q.predict({'actor':a})[0])
   p.reset();q.reset();self.assertEqual(p.reference_variation,0.);self.assertIsNone(p.reference_previous)
   np.testing.assert_array_equal(p.predict({'actor':a})[0],q.predict({'actor':a})[0])
 def test_actor_does_not_consume_teacher_contact_or_phase_labels(self):
  p,a=self.make_policy();x=p.predict({'actor':a})[0];p.reset();y=p.predict({'actor':a,'teacher_contact':np.full(4,1e5),'stage':999})[0]
  np.testing.assert_array_equal(x,y);self.assertTrue(np.isfinite(x).all());self.assertLessEqual(abs(x).max(),1.)
 def test_spatial_jet_preserves_a_quarter_circle_and_tangent(self):
  pose=torch.tensor([[[0.,.16,0.,1.,0.,0.,1.,0.,0.,0.,0.,.16,1.,0.,0.]]])
  result=transport_pose_jet(pose,torch.tensor([[[np.pi/2]]]))
  torch.testing.assert_close(result[0,0,:3],torch.tensor([0.,0.,.16]),atol=1e-7,rtol=0)
  torch.testing.assert_close(result[0,0,9:12],torch.tensor([0.,-.16,0.]),atol=1e-7,rtol=0)
  torch.testing.assert_close(torch.linalg.vector_norm(result[:,:,:3],dim=-1),torch.tensor([[.16]]),atol=1e-7,rtol=0)
 def test_zero_twist_and_zero_delta_are_finite_and_unchanged(self):
  pose=torch.randn(4,2,15);pose[:,:,3:9]=torch.tensor([1.,0.,0.,1.,0.,0.]);pose[:,:,12:15]=0.
  torch.testing.assert_close(transport_pose_jet(pose,torch.zeros(4,2,1)),pose,atol=0,rtol=0)
 def test_command_gate_refuses_missing_or_invalid_range_before_memory_update(self):
  p,a=self.make_policy();p.command_gate_enabled.fill_(True)
  for context in [None,[1.,-1.],[0.,float('nan')]]:
   observation={'actor':a}
   if context is not None:observation['command_range']=context
   with self.assertRaises(ValueError):p.predict(observation)
   self.assertIsNone(p.initial);self.assertEqual(p.phase,0.)
 def test_zero_neural_mixture_preserves_qualified_base_action_units(self):
  p,a=self.make_policy();p.command_gate_enabled.fill_(True)
  with torch.no_grad():
   for parameter in p.range_gate.parameters():parameter.zero_()
  actual=p.predict({'actor':a,'command_range':[0.,np.pi/2]})[0]
  with torch.no_grad():
   aa=torch.tensor(a)[None];expected=p.base.action(aa,p.initial[None],torch.tensor([p.phase]),torch.zeros(1))[0]
   expected[:12]*=torch.tensor(([.5]*3+[2/3]*3)*2)
  np.testing.assert_allclose(actual,expected.clamp(-1,1).numpy(),rtol=1e-6,atol=1e-7)
 def test_supplied_command_range_is_declared_and_matches_reference(self):
  from unified_steering.structured.driving_task import DrivingEnv,DrivingConfig
  env=DrivingEnv(DrivingConfig(command_range_observation=True))
  try:
   observation,_=env.reset(seed=48431,options={'free_side':-1,'exact_curriculum':True,'direction':1,'template':'short'})
   self.assertEqual(observation['command_range'].shape,(2,));self.assertIn('command_range',env.observation_space.spaces)
   np.testing.assert_allclose(observation['command_range'],[min(env.command_angles),max(env.command_angles)],rtol=1e-6)
  finally:env.close()
 def test_anchor_orientation_does_not_tilt_wheel_position_or_axis(self):
  from unified_steering.structured.driving_frame import adapt_pose,canonical_wrist_state
  from unified_steering.structured.driving_pose_policy import exp_rotation,rotation6
  p,a=self.make_policy();source=p.canonical_initial.clone();initial=source.clone();fk=initial[:18].reshape(2,9);turn=exp_rotation(torch.tensor([[.3,0.,0.],[.3,0.,0.]]));tilt=exp_rotation(torch.tensor([[0.,.25,0.],[0.,.25,0.]]));pos=torch.einsum('hij,hj->hi',turn,fk[:,:3]*.2);pos[:,0]+=.004;pos[:,1:]*=.165/.16;rot=turn@tilt@rotation6(fk[:,3:]);fk[:,:3]=pos/.2;fk[:,3:]=rot[:,:,:2].reshape(2,6)
  pose=p.values[0,:30].reshape(2,15).clone();pose[:,9:12]=torch.cross(torch.tensor([[1.,0.,0.],[1.,0.,0.]]),pose[:,:3],dim=-1);pose[:,12]=1.
  out=adapt_pose(pose,initial,source)
  torch.testing.assert_close(out[:,0],torch.full((2,),.004),atol=1e-7,rtol=0)
  torch.testing.assert_close(torch.linalg.vector_norm(out[:,1:3],dim=-1),torch.full((2,),.165),atol=1e-7,rtol=0)
  torch.testing.assert_close(out[:,12:15],torch.tensor([[1.,0.,0.],[1.,0.,0.]]),atol=1e-7,rtol=0)
  observed=torch.tensor(a);observed[102:120]=fk.ravel();restored=canonical_wrist_state(observed,initial,source)
  torch.testing.assert_close(restored,source[:18].reshape(2,9),atol=3e-7,rtol=0)
 def test_learned_anchor_update_changes_only_internal_memory(self):
  p,a=self.make_policy();p.anchor_gate_enabled.fill_(True)
  with torch.no_grad():
   for parameter in p.anchor_gate.parameters():parameter.zero_()
   p.anchor_gate[-1].bias[1]=10.
  saved=a.copy();p.predict({'actor':a});self.assertEqual(p.internal_rebase_count,1);np.testing.assert_array_equal(a,saved)
  self.assertEqual(p.reference_variation,0.);self.assertEqual(p.phase,0.)
 def test_wrist_anchor_renewal_does_not_accumulate_grasp_preload(self):
  p,a=self.make_policy();p.anchor_gate_enabled.fill_(True);p.renew_hand_anchor.fill_(False)
  p.values[:,30:]+=.15
  with torch.no_grad():
   for parameter in p.anchor_gate.parameters():parameter.zero_()
   p.anchor_gate[-1].bias[1]=10.
  p.predict({'actor':a});target=p.previous_intent[30:].clone();original_hand=p.initial[-12:].clone()
  for cycle in range(3):
   a[14:26]=target.numpy();a[102]+=.01;saved=a.copy()
   p.predict({'actor':a})
   torch.testing.assert_close(p.initial[-12:],original_hand,atol=0,rtol=0)
   torch.testing.assert_close(p.previous_intent[30:],target,atol=0,rtol=0)
   np.testing.assert_array_equal(a,saved)
   torch.testing.assert_close(p.initial[:18],torch.tensor(a[102:120]),atol=0,rtol=0)
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'teacher.zip';PathBasisModel(p).save(path)
   self.assertFalse(PathBasisModel.load(path).policy.renew_hand_anchor)
 def test_legacy_anchor_checkpoint_keeps_its_original_behavior(self):
  p,a=self.make_policy();state=p.state_dict();del state['renew_hand_anchor']
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'teacher.zip';torch.save(state,path);old=PathBasisModel.load(path).policy
   self.assertTrue(old.renew_hand_anchor)
   np.testing.assert_array_equal(p.predict({'actor':a})[0],old.predict({'actor':a})[0])
 def test_large_angle_memory_bound_preserves_base_limit(self):
  p,a=self.make_policy();self.assertEqual(float(p.phase_bound),np.pi/6)
  p.phase_bound.fill_(np.deg2rad(40.));p.base.velocity_head.bias.data.fill_(100.)
  p.predict({'actor':a});self.assertAlmostEqual(p.phase,np.deg2rad(40.),places=12)
  p.reset();p.command_gate_enabled.fill_(True)
  with torch.no_grad():
   for parameter in p.range_gate.parameters():parameter.zero_()
  p.predict({'actor':a,'command_range':[0.,np.pi/2]})
  self.assertAlmostEqual(p.phase,np.pi/6,places=12)
if __name__=='__main__':unittest.main()
