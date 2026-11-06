"""Small subprocess vector runner with explicit final observations and resets."""
import copy
import multiprocessing as mp
import traceback

import numpy as np

from .env import SteeringEnv


class TaskSchedule:
    """Balanced reset-time blocks; never switches a controller during a task."""
    def __init__(self, rng, config):
        self.rng,self.config,self.pending=rng,config,[]

    def next(self):
        if self.config.get('mode')!='single_direction':return None
        if not self.pending:
            counts=self.config['balanced_episode_counts']
            self.pending=list(self.rng.permutation([kind for kind,count in counts.items() for _ in range(count)]))
        return self.pending.pop()


def reset_task(env, rng, config, seed=None, task_kind=None):
    obs, info = env.reset(seed=seed)
    theta = info['initial_angle_rad']
    if config.get('mode')=='single_direction':
        if task_kind not in ('hold','positive','negative'):raise ValueError('Explicit balanced task kind required')
        task=task_kind
        direction={'hold':0,'positive':1,'negative':-1}[task]
        amplitude=direction*np.deg2rad(rng.uniform(*config['amplitude_range_deg'])) if direction else 0.
        speed=float(rng.uniform(*config['speed_range_deg_s']))
        env.config['reference']['speed_deg_s']=speed
        angles=[theta,theta+amplitude] if direction else [theta]
    elif rng.random() < config['hold_fraction']:
        angles, task = [theta], 'hold'
    else:
        amplitude = np.deg2rad(config['amplitude_deg'])*rng.choice([-1, 1])
        angles, task = [theta, theta+amplitude, theta-amplitude, theta], 'roundtrip'
    env.set_reference(angles)
    if config.get('mode')=='single_direction':
        duration=env.reference.duration+config['final_hold_seconds'] if direction else config['hold_episode_seconds']
        env.config['episode_seconds']=float(np.ceil(duration/env.period-1e-8)*env.period)
    obs = env.encoder.build(env.reference, 0.)
    env.last_obs = obs
    return obs, {**info, 'task': str(task), 'reference_duration_seconds': env.reference.duration,
                 'target_offset_deg':float(np.rad2deg(angles[-1]-theta)),
                 'direction':int(np.sign(angles[1]-theta)) if len(angles)>1 else 0}


def _worker(connection, env_config, task_config, seed):
    try:
        env = SteeringEnv(env_config)
        rng = np.random.default_rng(seed)
        schedule=TaskSchedule(rng,task_config)
        obs, initial = reset_task(env, rng, task_config, seed,task_kind=schedule.next())
        episode_return, length = 0., 0
        connection.send(('ok', (obs, env.hold_action(), initial)))
        while True:
            command, payload = connection.recv()
            if command == 'close':
                break
            if command != 'step':
                raise ValueError(f'Unknown command: {command}')
            final_obs, reward, terminated, truncated, info = env.step(payload)
            active_task=initial['task']
            if not info['observation_valid']:
                raise FloatingPointError('Invalid physics observation; stop training instead of learning from repaired data')
            episode_return += reward
            length += 1
            episode = None
            if terminated or truncated:
                episode = dict(return_=episode_return, length=length, task=initial['task'],
                               terminated=terminated, truncated=truncated, reason=info['termination_reason'],
                               reference_completed=bool(info['reference_time_seconds'] >= initial['reference_duration_seconds']),
                               direction=initial['direction'],target_offset_deg=initial['target_offset_deg'],
                               **info['episode_metrics'])
                obs, initial = reset_task(env, rng, task_config,task_kind=schedule.next())
                episode_return, length = 0., 0
            else:
                obs = final_obs
            # final_obs belongs to the action just taken; obs may belong to reset.
            record = dict(task=active_task,executed_action=info['executed_action'], physics_steps=info['physics_steps'],
                          interventions=info['execution_interventions'], reward_terms=info['reward_terms'],
                          collision_peak_n=info['collision_peak_n'])
            connection.send(('ok', (obs, final_obs['critic'], reward, terminated, truncated, episode, record)))
    except EOFError:
        pass
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


class WorkerPool:
    def __init__(self, env_config, task_config, count, seed):
        if count < 1:
            raise ValueError('At least one environment is required')
        context = mp.get_context('spawn')
        self.connections, self.processes = [], []
        try:
            for index in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, copy.deepcopy(env_config), task_config, seed+index))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            initial = [self._receive(c) for c in self.connections]
            self.observations = self.stack([item[0] for item in initial])
            self.initial_action = initial[0][1]
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection):
        if not connection.poll(60):
            raise TimeoutError('Simulation worker did not respond within 60 seconds')
        status, value = connection.recv()
        if status != 'ok':
            raise RuntimeError(value)
        return value

    @staticmethod
    def stack(observations):
        return {key: np.stack([obs[key] for obs in observations]) for key in ('actor', 'critic')}

    def step(self, actions):
        for connection, action in zip(self.connections, actions, strict=True):
            connection.send(('step', action))
        values = [self._receive(c) for c in self.connections]
        self.observations = self.stack([item[0] for item in values])
        return (self.observations, np.stack([item[1] for item in values]),
                np.array([item[2] for item in values], dtype=np.float32),
                np.array([item[3] for item in values], dtype=bool),
                np.array([item[4] for item in values], dtype=bool),
                [item[5] for item in values], [item[6] for item in values])

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()
