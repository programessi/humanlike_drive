"""Small dense strictly-convex QP with explicit feasibility/KKT verification."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

@dataclass
class QPSolution:
    x:np.ndarray
    success:bool
    iterations:int
    reason:str
    primal_residual:float
    dual_residual:float

def solve_qp(hessian,gradient,lo,hi,rows=None,lower=None,start=None,max_iterations=300):
    h=np.asarray(hessian,float);g=np.asarray(gradient,float);n=len(g)
    bounds=np.vstack([np.eye(n),-np.eye(n)])
    limits=np.r_[lo,-np.asarray(hi)]
    if rows is not None and len(rows):bounds=np.vstack([bounds,rows]);limits=np.r_[limits,lower]
    if not all(np.isfinite(v).all() for v in [h,g,bounds,limits]) or np.any(np.asarray(lo)>np.asarray(hi)):
        return QPSolution(np.zeros(n),False,0,'invalid',np.inf,np.inf)
    x=None if start is None else np.asarray(start,float).copy()
    if x is None or np.min(bounds@x-limits)<-1e-8:
        feasible=linprog(np.zeros(n),A_ub=-bounds,b_ub=-limits,bounds=[(None,None)]*n,method='highs')
        if not feasible.success:return QPSolution(np.zeros(n),False,0,'infeasible',np.inf,np.inf)
        x=feasible.x
    active=list(np.flatnonzero(abs(bounds@x-limits)<1e-8))
    for iteration in range(max_iterations):
        a=bounds[active];grad=h@x+g
        if active:
            kkt=np.block([[h,-a.T],[a,np.zeros((len(active),len(active)))]])
            solution=np.linalg.lstsq(kkt,np.r_[-grad,np.zeros(len(active))],rcond=1e-11)[0]
            step,multipliers=solution[:n],solution[n:]
        else:step=np.linalg.solve(h,-grad);multipliers=np.zeros(0)
        if np.max(abs(step))<1e-9:
            if not active or np.min(multipliers)>=-1e-7:
                primal=float(max(0.,np.max(limits-bounds@x)))
                dual=float(np.max(abs(grad-a.T@multipliers))) if active else float(np.max(abs(grad)))
                ok=primal<1e-7 and dual<1e-6
                return QPSolution(x,ok,iteration+1,'optimal' if ok else 'residual',primal,dual)
            del active[int(np.argmin(multipliers))];continue
        movement=bounds@step;slack=bounds@x-limits
        mask=np.ones(len(bounds),bool);mask[active]=False
        candidates=np.flatnonzero(mask & (movement<-1e-10))
        alpha,blocking=1.,None
        for i in candidates:
            value=max(0.,slack[i])/(-movement[i])
            if value<alpha-1e-12:alpha,blocking=value,int(i)
        x=x+alpha*step
        if blocking is not None:active.append(blocking)
    return QPSolution(x,False,max_iterations,'iteration_limit',float(max(0.,np.max(limits-bounds@x))),np.inf)
