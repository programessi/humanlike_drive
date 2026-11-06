"""Coarse route instructions from a preplanned route and GPS pose.

Only a categorical straight/left/right instruction reaches the visual policy;
route coordinates never replace its predicted steering path.
"""
from __future__ import annotations

import math
import numpy as np


def turn_command(forward,lateral,threshold_degrees=20):
    angle=math.degrees(math.atan2(lateral,forward))
    return 1 if angle < -threshold_degrees else 2 if angle > threshold_degrees else 0


class RouteNavigator:
    def __init__(self,points,spacing=2.,lookahead=15.):
        self.points=np.asarray(points,dtype=float)
        if self.points.ndim!=2 or self.points.shape[1]!=2 or len(self.points)<2:
            raise ValueError("Navigation route needs at least two XY points")
        self.spacing=spacing;self.lookahead=lookahead;self.index=0
        self.arcs=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
        self.progress=0.
        self.offset=0.

    @classmethod
    def from_map(cls,road_map,location,metres=2000,spacing=2.):
        wp=road_map.get_waypoint(location)
        points=[]
        for _ in range(int(metres/spacing)):
            points.append([wp.transform.location.x,wp.transform.location.y])
            candidates=wp.next(spacing)
            if not candidates:break
            wp=min(candidates,key=lambda item:abs((item.transform.rotation.yaw-wp.transform.rotation.yaw+180)%360-180))
        return cls(points,spacing=spacing)

    def update(self,x,y):
        # A local search prevents jumping between distant crossings of a route.
        start=max(0,self.index-2);stop=min(len(self.points),self.index+11)
        position=np.array([x,y])
        a=self.points[start:stop-1];delta=self.points[start+1:stop]-a
        squared=np.maximum(np.square(delta).sum(axis=1),1e-9)
        fraction=np.clip(((position-a)*delta).sum(axis=1)/squared,0,1)
        projected=a+fraction[:,None]*delta
        distances=np.linalg.norm(projected-position,axis=1)
        nearest=start+int(np.argmin(distances))
        self.offset=float(distances[nearest-start])
        progress=self.arcs[nearest]+fraction[nearest-start]*(self.arcs[nearest+1]-self.arcs[nearest])
        self.progress=max(self.progress,float(progress))
        self.index=min(len(self.points)-2,int(np.searchsorted(self.arcs,self.progress,side="right")-1))

    def path_relative(self,x,y,yaw_degrees,distances):
        self.update(x,y)
        forward_arcs=self.progress+np.asarray(distances)
        points=np.column_stack([np.interp(forward_arcs,self.arcs,self.points[:,i]) for i in range(2)])
        delta=points-np.array([x,y])
        yaw=math.radians(yaw_degrees)
        forward=math.cos(yaw)*delta[:,0]+math.sin(yaw)*delta[:,1]
        lateral=-math.sin(yaw)*delta[:,0]+math.cos(yaw)*delta[:,1]
        return np.column_stack([forward,lateral])

    def command(self,x,y,yaw_degrees):
        forward,lateral=self.path_relative(x,y,yaw_degrees,[self.lookahead])[0]
        return turn_command(forward,lateral)
