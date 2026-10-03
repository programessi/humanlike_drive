import unittest
import numpy as np

from humanlike_drive.navigation import RouteNavigator, turn_command


class NavigationTests(unittest.TestCase):
    def test_categorical_turns_and_local_route_progress(self):
        self.assertEqual(turn_command(15,.5),0)
        self.assertEqual(turn_command(10,-10),1)
        self.assertEqual(turn_command(10,10),2)
        route=RouteNavigator([[x,0] for x in range(0,102,2)])
        self.assertEqual(route.command(10,0,0),0)
        self.assertEqual(route.index,5)
        # A small GPS lateral offset does not become a steering instruction.
        self.assertEqual(route.command(12,1,0),0)
        self.assertEqual(route.index,6)

    def test_left_and_right_circular_routes(self):
        theta=np.linspace(0,np.pi/2,17)
        points=np.column_stack([20*np.sin(theta),20*(1-np.cos(theta))])
        self.assertEqual(RouteNavigator(points).command(0,0,0),2)
        points[:,1]*=-1
        self.assertEqual(RouteNavigator(points).command(0,0,0),1)

    def test_future_points_follow_one_route_at_continuous_distances(self):
        route=RouteNavigator([[x,0] for x in range(0,102,2)])
        xy=route.path_relative(11.5,1,0,[3,6,9,12,15])
        np.testing.assert_allclose(xy[:,0],[3,6,9,12,15])
        np.testing.assert_allclose(xy[:,1],-1)
        self.assertAlmostEqual(route.offset,1)
