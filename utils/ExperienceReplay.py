"""
    Implementation of experience replay buffers:
        - vanilla experience replay
        - Prioritized experience replay

    Code is implemented based on openai baselines
"""
from abc import ABC

import numpy as np
import abc
import threading

import IPython.terminal.debugger as Debug


class ReplayBuffer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def add(self):
        pass

    @abc.abstractmethod
    def sample_batch(self, batch_size):
        pass


class DQNReplayBuffer(ReplayBuffer, ABC):
    """
        Vanilla experience replay
            - the transitions are sampled with repeated possibility
            - using list to store the data
    """

    def __init__(self, buffer_size):
        super(DQNReplayBuffer, self).__init__()
        # total size of the replay buffer
        self.total_size = buffer_size

        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, reward, next_obs, done):
        # create a tuple
        trans = (obs, act, reward, next_obs, done)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        # lists for transitions
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = [], [], [], [], []

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d = data
            # store to the list
            obs_list.append(np.array(obs, copy=False))
            actions_list.append(np.array(act, copy=False))
            rewards_list.append(np.array(reward, copy=False))
            next_obs_list.append(np.array(next_obs, copy=False))
            dones_list.append(np.array(d, copy=False))
        # return the sampled batch data as numpy arrays
        return np.array(obs_list), np.array(actions_list), np.array(rewards_list), np.array(next_obs_list), np.array(
            dones_list)

    def sample_batch(self, batch_size):
        # sample indices with replaced
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)


class GoalDQNReplayBuffer(ReplayBuffer, ABC):
    """
           Vanilla experience replay
               - the transitions are sampled with repeated possibility
               - using list to store the data
       """

    def __init__(self, buffer_size):
        super(GoalDQNReplayBuffer, self).__init__()
        # total size of the replay buffer
        self.total_size = buffer_size

        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, reward, next_obs, done, goal):
        # create a tuple
        goal = np.array(goal)
        trans = (obs, act, reward, next_obs, done, goal)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        # lists for transitions
        obs_list, actions_list, rewards_list, next_obs_list, dones_list, goal_list = [], [], [], [], [], []

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d, g = data
            # store to the list
            obs_list.append(np.array(obs, copy=False))
            actions_list.append(np.array(act, copy=False))
            rewards_list.append(np.array(reward, copy=False))
            next_obs_list.append(np.array(next_obs, copy=False))
            dones_list.append(np.array(d, copy=False))
            goal_list.append(np.array(g, copy=False))
        # return the sampled batch data as numpy arrays
        return np.array(obs_list), np.array(actions_list), np.array(rewards_list), np.array(next_obs_list), np.array(
            dones_list), np.array(goal_list)

    def sample_batch(self, batch_size):
        # sample indices with replaced
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)


""" Implement the HER replay buffer
    Note: The majority of the code is adopt from OpenAI baselines and this Github repo

    Brief explanation: 
        In both implementations, the idea is slightly different from the original paper, where HER will relabel the 
        transitions for each episode during data collection. In other words, the relabeled transitions will be stored
        in the buffer. However, in the following implementation, we store the episodes rather than transitions. We
        only use HER to relabel the sampled transitions. I guess there are the following benefits：
            - Relabeling the transitions during the data collection is cumbersome. Because the training will only use
              a batch of the data. Therefore, only relabeling the transitions in the sampled batch will be more
              efficient.
            - Storing the relabeled data in the replay buffer and then sample is less efficient. Because there is no
              guarantee that the relabeled data will be sampled. Therefore, relabeling the transitions in the sampled 
              batch is also efficient for training.

    Implementation details:
        The HER implementation consist of two main parts:
            - The SampleFuncReplayBuffer: This is a class that defines the data storage, adding and sampling methods.
              However, three things need to be clarified:
                - The replay buffer stores episodes with fixed length (i.e., the max episode length)
                - The replay buffer sample the transitions based on customized sampling function
                - The replay buffer maintain a non-circular data storing

            - The HER: This is a class that defines the HER algorithm with relabeling strategy, ratio, and sampler
              function.
"""


class SampleFuncReplayBuffer(object):
    # init the buffer
    def __init__(self, env_params, size_in_transitions, sample_func):
        # environment parameters
        self.env_params = env_params
        self.T = env_params['max_episode_step']

        # memory management
        self.size_in_transitions = size_in_transitions  # size of the buffer in transitions
        self.size_in_episodes = size_in_transitions // self.T  # size of the buffer in episodes

        self.stored_episode_num = 0  # current stored episodes
        self.stored_transitions_num = 0  # current stored transitions

        # obtain the shape of the observation (obs), achieved_goal (ag), desired_goal (dg), and action (act)
        # please note, in this implementation, the value in "obs" is the same as "ag"
        obs_shape = [*env_params['obs'].shape] if len(env_params['obs'].shape) > 1 else [env_params['obs'].shape[0]]
        ag_shape = [*env_params['ag'].shape] if len(env_params['ag'].shape) > 1 else [env_params['ag'].shape[0]]
        dg_shape = [*env_params['dg'].shape] if len(env_params['dg'].shape) > 1 else [env_params['dg'].shape[0]]
        act_shape = [*env_params['act'].shape] if len(env_params['act'].shape) > 1 else [env_params['act'].shape[0]]
        # buffer for episodes: size_in_episodes x (T or T + 1) x shape of the data
        self.buffers = {
            'obs': np.empty([self.size_in_episodes, self.T + 1, *obs_shape], dtype=np.float),
            'ag': np.empty([self.size_in_episodes, self.T + 1, *ag_shape], dtype=np.float),
            'dg': np.empty([self.size_in_episodes, self.T, *dg_shape], dtype=np.float),
            'act': np.empty([self.size_in_episodes, self.T, *act_shape], dtype=np.int)
        }  # data storage

        # sampler function
        self.sample_func = sample_func

    # get the size of the buffer, by default, we return the number of episodes
    def __len__(self):
        # with self.lock:
        return self.stored_episode_num

    # get the storage indices
    def _get_storage_idx(self, inc=None):
        size_inc = inc or 1  # size increment in terms of number of episodes
        assert size_inc <= self.size_in_episodes, "The batch size is too big!"

        # compute the indices for the available storage
        if self.stored_episode_num + size_inc <= self.size_in_episodes:  # there is enough room to add new data
            indices = np.arange(self.stored_episode_num, self.stored_episode_num + size_inc)
        elif self.stored_episode_num < self.size_in_episodes:  # the memory is not full but no room for all new data
            overflow_num = size_inc - (self.size_in_episodes - self.stored_episode_num)  # compute the overflow
            indices_a = np.arange(self.stored_episode_num, self.size_in_episodes)  # sequential storage
            indices_b = np.random.randint(0, self.stored_episode_num, overflow_num)  # random overwrite old episodes
            indices = np.concatenate([indices_a, indices_b])  # concatenate the indices
        else:
            indices = np.random.randint(0, self.stored_episode_num, size_inc)

        # update memory size
        self.stored_episode_num = np.min([self.size_in_episodes, self.stored_episode_num + size_inc])

        # return the indices
        if size_inc == 1:
            indices = indices.tolist()

        # note: the return should be list of indices for storage
        return indices

    # store the data
    def store_episode(self, episode_batch):  # size: batch x T or T+1 x shape of the data
        # split the data. Note, mb is an abbreviation of multi-batch for multi-threads data collection
        mb_obs, mb_ag, mb_dg, mb_act = episode_batch
        # get the batch size
        batch_size = mb_obs.shape[0]

        # with self.lock:
        # get the available indices (empty or need to be rewritten) to store the data
        indices = self._get_storage_idx(batch_size)

        # store the data using the indices: an interesting implementation of circle buffer
        self.buffers['obs'][indices] = mb_obs
        self.buffers['ag'][indices] = mb_ag
        self.buffers['dg'][indices] = mb_dg
        self.buffers['act'][indices] = mb_act

        # increase the transition counter
        self.stored_transitions_num += batch_size * self.T
        self.stored_transitions_num = np.min([self.size_in_transitions, self.stored_transitions_num])

    # sample a batch data
    def sample(self, batch_size):
        temp_buffers = {}  # a temporal memory buffer to avoid changing the raw one
        # with self.lock:
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.stored_episode_num]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]  # add next observation
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]  # add next achieved goal

        # sample the transitions
        transitions = self.sample_func(temp_buffers, batch_size)

        return transitions


""" HER for images as observations

"""


class SampleFuncReplayBufferImages(object):
    # init the buffer
    def __init__(self, env_params, size_in_transitions, sample_func):
        # environment parameters
        self.env_params = env_params
        self.T = env_params['max_episode_step']

        # memory management
        self.size_in_transitions = size_in_transitions  # size of the buffer in transitions
        self.size_in_episodes = size_in_transitions // self.T  # size of the buffer in episodes

        self.stored_episode_num = 0  # current stored episodes
        self.stored_transitions_num = 0  # current stored transitions

        # obtain the shape of the observation (obs), achieved_goal (ag), desired_goal (dg), and action (act)
        # please note, in this implementation, the value in "obs" is the same as "ag"
        obs_shape = np.array(env_params['obs']).shape
        ag_shape = np.array(env_params['ag']).shape
        dg_shape = np.array(env_params['dg']).shape
        act_shape = np.array(env_params['act']).shape
        # buffer for episodes: size_in_episodes x (T or T + 1) x shape of the data
        # (T + 1) is because we store o_0 to o_T-1 + o_T
        self.buffers = {
            'obs': np.empty([self.size_in_episodes, self.T + 1, *obs_shape], dtype=np.float32),
            'ag': np.empty([self.size_in_episodes, self.T + 1, *ag_shape], dtype=np.float32),
            'dg': np.empty([self.size_in_episodes, self.T, *dg_shape], dtype=np.float32),
            'act': np.empty([self.size_in_episodes, self.T, *act_shape], dtype=np.int32)
        }  # data storage

        # sampler function
        self.sample_func = sample_func

    # get the size of the buffer, by default, we return the number of episodes
    def __len__(self):
        # with self.lock:
        return self.stored_episode_num

    # get the storage indices
    def _get_storage_idx(self, inc=None):
        size_inc = inc or 1  # size increment in terms of number of episodes
        assert size_inc <= self.size_in_episodes, "The batch size is too big!"

        # compute the indices for the available storage
        if self.stored_episode_num + size_inc <= self.size_in_episodes:  # there is enough room to add new data
            indices = np.arange(self.stored_episode_num, self.stored_episode_num + size_inc)
        elif self.stored_episode_num < self.size_in_episodes:  # the memory is not full but no room for all new data
            overflow_num = size_inc - (self.size_in_episodes - self.stored_episode_num)  # compute the overflow
            indices_a = np.arange(self.stored_episode_num, self.size_in_episodes)  # sequential storage
            indices_b = np.random.randint(0, self.stored_episode_num, overflow_num)  # random overwrite old episodes
            indices = np.concatenate([indices_a, indices_b])  # concatenate the indices
        else:
            indices = np.random.randint(0, self.stored_episode_num, size_inc)

        # update memory size
        self.stored_episode_num = np.min([self.size_in_episodes, self.stored_episode_num + size_inc])

        # return the indices
        if size_inc == 1:
            indices = indices.tolist()

        # note: the return should be list of indices for storage
        return indices

    # store the data
    def store_episode(self, episode_batch):  # size: batch x T or T+1 x shape of the data
        # split the data. Note, mb is an abbreviation of multi-batch for multi-threads data collection
        mb_obs, mb_ag, mb_dg, mb_act = episode_batch
        # get the batch size
        batch_size = mb_obs.shape[0]

        # with self.lock:
        # get the available indices (empty or need to be rewritten) to store the data
        indices = self._get_storage_idx(batch_size)

        # store the data using the indices: an interesting implementation of circle buffer
        self.buffers['obs'][indices] = mb_obs
        self.buffers['ag'][indices] = mb_ag
        self.buffers['dg'][indices] = mb_dg
        self.buffers['act'][indices] = mb_act

        # increase the transition counter
        self.stored_transitions_num += batch_size * self.T
        self.stored_transitions_num = np.min([self.size_in_transitions, self.stored_transitions_num])

    # sample a batch data
    def sample(self, batch_size):
        temp_buffers = {}  # a temporal memory buffer to avoid changing the raw one
        # with self.lock:
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.stored_episode_num]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :, :, :, :]  # add next observation
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :, :, :, :]  # add next achieved goal

        # sample the transitions
        transitions = self.sample_func(temp_buffers, batch_size)

        return transitions









