"""Deterministic stratification across workers, without execution-time roles."""
import numpy as np


# Same support proportions .25/.375/.375; each block has both sequence signs.
CONDITIONS=(("hold",5.,1),("step",5.,1),("step",5.,-1),
            ("step",10.,1),("step",10.,-1),("sequence",10.,1),
            ("sequence",10.,-1),("sine",5.,1))


def balanced_condition(index, seed=7):
    block,slot=divmod(index,64)
    entries=[(free,mode,amplitude,sign) for free,repetitions in ((-1,2),(1,3),(0,3))
             for _ in range(repetitions) for mode,amplitude,sign in CONDITIONS]
    order=np.random.default_rng(seed+104729*block).permutation(64)
    free,mode,amplitude,sign=entries[order[slot]]
    if mode=="sine":
        amplitude=5. if block%2==0 else 10.
        sign=1 if block%2==0 else -1
    return dict(free_side=free,mode=mode,training_amplitude_deg=amplitude,direction=sign,
                course_block=block,course_slot=slot)
