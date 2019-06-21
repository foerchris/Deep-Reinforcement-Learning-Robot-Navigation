# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs_1, obs_2, obs_3, reward, done, info = env.step(data)
            if done:
                obs_1, obs_2, obs_3 = env.reset()
            remote.send((obs_1, obs_2, obs_3, reward, done, info))
        elif cmd == 'reset':
            obs_1, obs_2, obs_3 = env.reset()
            remote.send((obs_1, obs_2, obs_3))
        elif cmd == 'get_state':
            obs_1, obs_2, obs_3 = env.get_state()
            remote.send((obs_1, obs_2, obs_3))
        elif cmd == 'reset_task':
            obs_1, obs_2, obs_3 = env.reset_task()
            remote.send((obs_1, obs_2, obs_3))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'set_episode_length':
            env.set_episode_length(data)

        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass
    def get_states(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs_1, obs_2, obs_3, rews, dones, infos):
         - obs_1, obs_2, obs_3: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def reset_test(self, number):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_test(self, actions, number):
        self.step_async_test(actions,number)
        return self.step_wait_test(number)

    def step_async_test(self, actions, number):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait_test(self, number):
        """
        Wait for the step taken with step_async().
        Returns (obs_1, obs_2, obs_3, rews, dones, infos):
         - obs_1, obs_2, obs_3: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_test(self):
        """
        Clean up the environments' resources.
        """
        pass




    def set_episode_length(self, length):
        """
        Clean up the environments' resources.
        """
        pass

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def get_states(self):
        for remote in self.remotes:
            remote.send(('get_state', None))
        results = [remote.recv() for remote in self.remotes]
        obs_1, obs_2, obs_3 = zip(*results)
        return np.stack(obs_1), np.stack(obs_2), np.stack(obs_3)

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_1, obs_2, obs_3, rews, dones, infos = zip(*results)
        return np.stack(obs_1), np.stack(obs_2), np.stack(obs_3), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs_1, obs_2, obs_3 = zip(*results)
        return np.stack(obs_1), np.stack(obs_2), np.stack(obs_3)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        results = [remote.recv() for remote in self.remotes]
        obs_1, obs_2, obs_3 = zip(*results)
        return np.stack(obs_1), np.stack(obs_2), np.stack(obs_3)


    def step_async_test(self, actions, number):
        if (number > 0 and number <= self.num_envs):
            for remote, action in zip(self.remotes[number-1:number], actions[number-1:number]):
                remote.send(('step', action))
            self.waiting = True
        else:
            print('multi_env step_async_test error, number: ' + str(number) + ' is out of range')


    def step_wait_test(self, number):
        if (number > 0 and number <= self.num_envs):
            results = [remote.recv() for remote in self.remotes[number-1:number]]
            self.waiting = False
            obs_1, obs_2, obs_3, rews, dones, infos = zip(*results)
            return np.stack(obs_1), np.stack(obs_2), np.stack(obs_3), np.stack(rews), np.stack(dones), infos
        else:
            print('multi_env step_wait_test error, number: ' + str(number) + ' is out of range')


    def reset_test(self, number):
        if(number > 0 and number <=self.num_envs):
            for remote in self.remotes[number-1:number]:
                remote.send(('reset', None))
            results = [remote.recv() for remote in self.remotes[number-1:number]]
            obs_1, obs_2, obs_3 = zip(*results)
            return np.stack(obs_1), np.stack(obs_2), np.stack(obs_3)
        else:
            print('multi_env reset_test error, number: ' + str(number) + ' is out of range')

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def set_episode_length(self, length):
        for remote, length in zip(self.remotes, length):
            remote.send(('set_episode_length', length))

    def __len__(self):
        return self.nenvs