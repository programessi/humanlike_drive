import unittest
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from unified_steering.structured.box_qp import solve_qp

class BoxQPTests(unittest.TestCase):
    def test_random_feasible_qps_match_independent_scipy_optimum(self):
        rng=np.random.default_rng(805)
        for _ in range(12):
            a=rng.normal(size=(28,14));h=a.T@a+.01*np.eye(14);g=rng.normal(size=14)
            rows=rng.normal(size=(8,14));lower=-rng.uniform(.02,.1,8);lo=-np.ones(14);hi=np.ones(14)
            result=solve_qp(h,g,lo,hi,rows,lower)
            self.assertTrue(result.success,result)
            scipy=minimize(lambda x:.5*x@h@x+g@x,np.zeros(14),jac=lambda x:h@x+g,
                bounds=list(zip(lo,hi)),constraints=[LinearConstraint(rows,lower,np.inf)],method='SLSQP',options=dict(ftol=1e-11,maxiter=300))
            self.assertTrue(scipy.success)
            self.assertLess(abs((.5*result.x@h@result.x+g@result.x)-scipy.fun),1e-7)
            self.assertGreaterEqual(min(rows@result.x-lower),-1e-7)
    def test_true_infeasibility_is_never_accepted(self):
        r=solve_qp(np.eye(2),np.zeros(2),[-1,-1],[1,1],[[1,0]],[2])
        self.assertFalse(r.success);self.assertEqual(r.reason,'infeasible')

if __name__=='__main__':unittest.main()
