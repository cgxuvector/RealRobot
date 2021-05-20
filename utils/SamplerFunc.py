"""
    This script contains the implementation of sampler functions
"""
import numpy as np
import IPython.terminal.debugger as Debug


ACTION_NAME = ['left', 'down', 'right', 'up']


class HER(object):
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        # replay strategy
        self.replay_strategy = replay_strategy
        # number of the replay
        self.replay_k = replay_k
        # reward function
        self.reward_func = reward_func

        # set replay ratio
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1.0 / (1 + replay_k))
        else:
            self.future_p = 0

    def sample_transitions(self, episode_batch, batch_size_in_transitions):
        """
        Logic: the function is going to sample one transition from each episode.
            1. Sample #(batch_size_in_transitions) episodes (implemented by randomly sampling indices)
            2. Sample #(batch_size_in_transitions) transitions in each episode (same)
            3. Sample the HER indices among the sampled indices
            4. Filter out the HER indices and do the relabeling stuff
            5. Return the sampled batch data

        :param episode_batch: dictionary (same like the self.buffers in replay buffer) that contains all stored episodes
        :param batch_size_in_transitions: batch size in transitions
        :return:
        """
        # get the size of the episode
        T = episode_batch['act'].shape[1]
        # get the available episode number
        rollout_batch_size = episode_batch['act'].shape[0]
        # get the batch size
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to use
        """ Sample the episodes
        """
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)  # sample #(batch_size) episodes
        """ Sample the timesteps
        """
        t_samples = np.random.randint(T, size=batch_size)  # each episode sample #(batch_size) transitions
        # extract the transitions based on episode id and time step id
        """ Extract the transitions #(batch_size_in_transitions)
        """
        transitions = {key: episode_batch[key][episode_idxs, t_samples] for key in episode_batch.keys()}

        # # print out the sampled transitions
        # for i in range(batch_size_in_transitions):
        #     print(f"state={transitions['obs'][i].tolist()},"
        #           f" act={ACTION_NAME[transitions['act'][i].tolist()[0]]},"
        #           f" next_obs={transitions['obs_next'][i].tolist()},"
        #           f" goal={transitions['dg'][i].tolist()},"
        #           f" ag={transitions['ag'][i].tolist()}")

        """ Sample the HER indices in the sampled batch indices
        """
        # her indices indicates which transition should be relabeled in the sampled transitions batch
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        """ Sample the future indices for all the sampled transitions: using sampling random offset
        """
        # compute the future offset for all the transitions in the batch
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        """ Extract the transitions that need to be relabeled
        """
        # get the indices of the future transitions that need to be relabeled
        future_indices = (t_samples + 1 + future_offset)[her_indices]

        """ HER: Relabeling happens
        """
        # replace the goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indices], future_indices]
        # print(future_ag)
        transitions['dg'][her_indices] = future_ag

        # print("After relabeling the achieved goals")
        #
        # for i in range(batch_size_in_transitions):
        #     print(f"state={transitions['obs'][i].tolist()},"
        #           f" act={ACTION_NAME[transitions['act'][i].tolist()[0]]},"
        #           f" next_obs={transitions['obs_next'][i].tolist()},"
        #           f" goal={transitions['dg'][i].tolist()},"
        #           f" ag={transitions['ag'][i].tolist()}")

        # compute the reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['dg']), 1)

        # print("After compute the rewards")
        #
        # for i in range(batch_size_in_transitions):
        #     print(f"state={transitions['obs'][i].tolist()},"
        #           f" act={ACTION_NAME[transitions['act'][i].tolist()[0]]},"
        #           f" reward={transitions['r'][i]}"
        #           f" next_obs={transitions['obs_next'][i].tolist()},"
        #           f" goal={transitions['dg'][i].tolist()},"
        #           f" ag={transitions['ag'][i].tolist()}")
        #
        # Debug.set_trace()

        # re-shape the transitions
        # transitions = {key: transitions[key].reshape(batch_size, *transitions[key].shape[1:]) for key in transitions.keys()}

        # re-arrange to adapt to my implementation
        transitions_rearranged = {
            'obs': transitions['obs'],
            'action': transitions['act'],
            'reward': transitions['r'],
            'next_obs': transitions['obs_next'],
            'goal': transitions['dg']
        }

        return transitions_rearranged

    def sample_transitions_images(self, episode_batch, batch_size_in_transitions):
        """
        Logic: the function is going to sample one transition from each episode.
            1. Sample #(batch_size_in_transitions) episodes (implemented by randomly sampling indices)
            2. Sample #(batch_size_in_transitions) transitions in each episode (same)
            3. Sample the HER indices among the sampled indices
            4. Filter out the HER indices and do the relabeling stuff
            5. Return the sampled batch data

        :param episode_batch: dictionary (same like the self.buffers in replay buffer) that contains all stored episodes
        :param batch_size_in_transitions: batch size in transitions
        :return:
        """
        # get the size of the episode
        T = episode_batch['act'].shape[1]
        # get the available episode number
        rollout_batch_size = episode_batch['act'].shape[0]
        # get the batch size
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to use
        """ Sample the episodes
        """
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)  # sample #(batch_size) episodes
        """ Sample the timesteps
        """
        t_samples = np.random.randint(T, size=batch_size)  # each episode sample #(batch_size) transitions
        # extract the transitions based on episode id and time step id
        """ Extract the transitions #(batch_size_in_transitions)
        """
        transitions = {key: episode_batch[key][episode_idxs, t_samples] for key in episode_batch.keys()}

        """ Sample the HER indices in the sampled batch indices
        """
        # her indices indicates which transition should be relabeled in the sampled transitions batch
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)[0].tolist()
        """ Sample the future indices for all the sampled transitions: using sampling random offset
        """
        # compute the future offset for all the transitions in the batch
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        """ Extract the transitions that need to be relabeled
        """
        # get the indices of the future transitions that need to be relabeled
        future_indices = (t_samples + 1 + future_offset)[her_indices]

        """ HER: Relabeling happens
        """
        # replace the goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indices], future_indices]
        # print(future_ag)
        transitions['dg'][her_indices] = future_ag

        # compute the reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['dg']), 1)

        # re-arrange to adapt to my implementation
        transitions_rearranged = {
            'obs': transitions['obs'],
            'action': transitions['act'],
            'reward': transitions['r'],
            'next_obs': transitions['obs_next'],
            'goal': transitions['dg']
        }

        return transitions_rearranged
