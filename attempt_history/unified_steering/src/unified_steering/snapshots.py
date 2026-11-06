"""Reset-free cloning for causal probes: physical integration, executor and task state."""
import copy

import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def capture(env):
    state = np.zeros(mujoco.mj_stateSize(env.sim.model,STATE_SPEC))
    mujoco.mj_getState(env.sim.model,env.sim.data,state,STATE_SPEC)
    return dict(integration=state,warmstart=env.sim.data.qacc_warmstart.copy(),
                executor=copy.deepcopy(vars(env.executor)),reference=copy.deepcopy(env.reference),
                reference_origin=env.reference_origin,steps=env.steps,done=env.done,
                metrics=copy.deepcopy(env.metrics),last_obs=copy.deepcopy(env.last_obs),
                last_load=copy.deepcopy(env.sim.last_wheel_load),rng=copy.deepcopy(env.np_random.bit_generator.state))


def restore(env, snapshot):
    m,d = env.sim.model,env.sim.data
    mujoco.mj_resetData(m,d)
    mujoco.mj_setState(m,d,snapshot['integration'],STATE_SPEC)
    mujoco.mj_forward(m,d)
    d.qacc_warmstart[:] = snapshot['warmstart']
    env.executor.__dict__ = copy.deepcopy(snapshot['executor'])
    for name in ('reference','reference_origin','steps','done','metrics','last_obs'):
        setattr(env,name,copy.deepcopy(snapshot[name]))
    env.sim.last_wheel_load = copy.deepcopy(snapshot['last_load'])
    env.np_random.bit_generator.state = copy.deepcopy(snapshot['rng'])
