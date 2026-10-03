"""Verify waypoint network gradients and geometry sign conventions."""
import unittest
import numpy as np
import torch
from humanlike_drive.vision import WaypointPolicy, image_tensor, LOOKAHEAD_METRES, policy_from_checkpoint
from humanlike_drive.carla_bridge import path_to_steer, safe_speed_target, residual_scale, arc_waypoints_to_xy


class VisionTests(unittest.TestCase):
    def test_forward_and_backprop(self):
        torch.set_num_threads(1)
        model=WaypointPolicy()
        rgb=image_tensor(np.zeros((360,640,3),dtype=np.uint8))[None]
        prediction=model(rgb,torch.tensor([3.]),torch.tensor([[1.,0.,0.]]))
        self.assertEqual(tuple(prediction.shape),(1,5))
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())

    def test_spatial_xy_policy_checkpoint_and_gradients(self):
        torch.set_num_threads(1)
        original=WaypointPolicy()
        model=WaypointPolicy(spatial_pool=(2,4),predict_xy=True)
        model.warm_start({"state_dict":original.state_dict()})
        rgb=image_tensor(np.zeros((360,640,3),dtype=np.uint8))[None]
        inputs=(rgb,torch.tensor([3.]),torch.tensor([[1.,0.,0.]]))
        output=model(*inputs)
        self.assertEqual(tuple(output.shape),(1,5,2))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head[0].weight.grad).all())
        clone=policy_from_checkpoint({"state_dict":model.state_dict(),"model_config":{"spatial_pool":[2,4],"predict_xy":True}})
        model.eval();clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(*inputs),clone(*inputs))

    def test_navigation_branch_receives_selected_command_gradients(self):
        torch.set_num_threads(1)
        base=WaypointPolicy(spatial_pool=(2,4),predict_xy=True)
        model=WaypointPolicy(spatial_pool=(2,4),predict_xy=True,navigation_branches=True)
        model.warm_start({"state_dict":base.state_dict()})
        rgb=image_tensor(np.zeros((360,640,3),dtype=np.uint8))[None]
        output=model(rgb,torch.tensor([3.]),torch.tensor([[0.,0.,1.]]))
        output.square().mean().backward()
        self.assertGreater(float(model.branches[2][-1].weight.grad.norm()),0)
        self.assertEqual(float(model.branches[0][-1].weight.grad.norm()),0)
        self.assertTrue(torch.isfinite(model.head[0].weight.grad).all())

    def test_path_to_steer(self):
        xy=np.column_stack([LOOKAHEAD_METRES,np.zeros(5)])
        self.assertEqual(path_to_steer(xy,np.deg2rad(35)),0)
        xy[:,1]=1
        self.assertGreater(path_to_steer(xy,np.deg2rad(35)),0)
        xy[:,1]=-1
        self.assertLess(path_to_steer(xy,np.deg2rad(35)),0)

    def test_speed_reduces_for_turn_and_actuator_lag(self):
        self.assertEqual(safe_speed_target(3,0,0),3)
        self.assertLess(safe_speed_target(3,.1,0),3)
        self.assertLess(safe_speed_target(3,.1,.3),safe_speed_target(3,.1,0))
        self.assertEqual(safe_speed_target(3,.1,.3),safe_speed_target(3,-.1,-.3))
        self.assertGreaterEqual(safe_speed_target(3,1,1),.6)

    def test_guard_falls_back_outside_trained_motion(self):
        self.assertEqual(residual_scale(0,0),1)
        self.assertEqual(residual_scale(0,np.deg2rad(20)),0)
        self.assertEqual(residual_scale(np.deg2rad(50),np.deg2rad(50)),0)
        self.assertGreater(residual_scale(np.deg2rad(10),0),0)
        self.assertLess(residual_scale(np.deg2rad(10),0),1)

    def test_road_arc_labels_recover_a_circular_path(self):
        distances=np.array(LOOKAHEAD_METRES)
        radius=10
        angle=distances/radius
        y=radius*(1-np.cos(angle))
        xy=arc_waypoints_to_xy(y)
        np.testing.assert_allclose(xy[:,0],radius*np.sin(angle),atol=1e-7)
        self.assertAlmostEqual(path_to_steer(xy,1),np.arctan(2.8/radius),places=7)
        np.testing.assert_allclose(arc_waypoints_to_xy(np.zeros(5))[:,0],distances)


if __name__=="__main__":unittest.main()
