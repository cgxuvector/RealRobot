from utils.Schedule import LinearSchedule
from utils.ExperienceReplay import GoalDQNReplayBuffer
from utils.ExperienceReplay import SampleFuncReplayBuffer, SampleFuncReplayBufferImages
from utils.SamplerFunc import HER
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import torch
from datetime import datetime
import os

import IPython.terminal.debugger as Debug

ACTION_NAME = ['turn_left', 'turn_right', 'forward', 'backward']


class GoalDQNExperiment(object):
    def __init__(self, agent, env, test_env, trn_params):
        # initialize the experiment
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.trn_params = trn_params

        # training parameters
        self.use_obs = trn_params['use_obs']
        self.schedule = LinearSchedule(1, 0.01, trn_params['total_time_steps'] / 2)
        self.use_her = trn_params['use_her']
        self.memory = GoalDQNReplayBuffer(trn_params['memory_size'])
        self.start_train_step = trn_params['start_train_step']
        self.total_time_steps = trn_params['total_time_steps']
        self.update_policy_freq = trn_params['update_policy_freq']
        self.update_target_freq = trn_params['update_target_freq']
        self.memory_size = trn_params['memory_size']
        self.batch_size = trn_params['batch_size']

        # special modifications
        self.use_her = trn_params['use_her']  # whether use her

        # save results
        self.trn_returns = []
        self.eval_success = []
        self.save_dir = trn_params['save_dir']
        self.model_name = trn_params['model_name']

        # create the summary writer
        current_time = datetime.now()
        date = current_time.strftime("%m-%d")
        time = current_time.strftime("%H-%M-%S")
        label = self.trn_params['run_label']
        target_log_dir = os.path.join('./runs',
                                      label,
                                      date,
                                      f"{time}_{trn_params['env_name'].split('.')[0]}")

        # create the summary writer
        self.tb = SummaryWriter(log_dir=target_log_dir,
                                comment=f"_label={self.trn_params['run_label']}_"
                                        f"{self.trn_params['env_name'].split('.')[0]}_"
                                        f"{self.trn_params['batch_size']}_"
                                        f"{self.trn_params['update_target_freq']}")

    def reset(self):
        return self.env.reset()

    def get_action(self, obs):
        return self.agent.get_action(obs)

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        return next_obs, reward, done, {}

    def eval(self):
        # number of the evaluation
        eval_total_num = 100
        success_num = 0.0
        optimal_steps = int(round(self.env.room_size / self.env.forward_step_size) + 1)
        # start evaluation
        for i in range(eval_total_num):
            # reset the environment
            obs = self.reset()
            for t in range(optimal_steps):
                # get action
                self.agent.eps = 0
                action = self.get_action(obs)
                # step in the environment
                next_obs, reward, done, _ = self.step(action)

                if reward == 0:
                    success_num += 1
                    break
                else:
                    obs = next_obs

        # record the success
        self.eval_success.append(success_num / eval_total_num)

    def save(self):
        pass

    def save_model(self, save_dir):
        torch.save(self.agent.behavior_policy_net.state_dict(),
                   save_dir + '/model/' + self.trn_params['model_name'] + '.pt')

    def save_data(self, save_dir):
        np.save(f'./{save_dir}/data/trn_return.npy', self.trn_returns)
        np.save(f'./{save_dir}/data/trn_acc.npy', self.eval_success)

    def run(self):
        # training variables
        episode_t = 0
        episode_idx = 0
        rewards = []

        # create the saving directory
        current_time = datetime.today()
        run_mode = os.path.join(self.trn_params['run_label'],
                                current_time.strftime("%m-%d"),
                                current_time.strftime("%H-%M-%S") +
                                f"_{self.trn_params['run_label']}_{self.trn_params['env_name'].split('.')[0]}")
        results_dir = os.path.join(self.trn_params['save_dir'], run_mode)

        # remove the old one
        if os.path.exists(results_dir):
            os.system(f"rm -r {results_dir}")
        # create the new one
        os.makedirs(f"{results_dir}/model")
        os.makedirs(f"{results_dir}/data")

        # reset the environment
        obs = self.env.reset()

        # start training
        pbar = tqdm.trange(self.total_time_steps)
        start_pos = obs['observation']
        goal_pos = obs['desired_goal']
        last_eval_success = 0
        for t in pbar:
            # get one action
            self.agent.eps = self.schedule.get_value(t)
            action = self.get_action(obs)
            # step in the environment
            next_obs, reward, done, _ = self.step(action)

            # only done for the true goal
            done = True if reward == 0 else False

            # add to the buffer
            self.memory.add(obs['observation'], action, reward, next_obs['observation'], done, obs['desired_goal'])
            rewards.append(reward)

            # if not self.use_obs:
            #     print(f" step = {t}"
            #           f" state={obs['observation']},"
            #           f" act={action},"
            #           f" next_state={next_obs['observation']},"
            #           f" reward={reward}, done={done},"
            #           f" goal={obs['desired_goal']}")
            # else:
            #     print(f" step = {t}, "
            #           f" agent pos = {self.env.agent.pos},"
            #           f" goal pos = {self.env.goal_info['pos']}, "
            #           f" distance = {self.env.goal_info['dist']}, "
            #           f" done={done},"
            #           f" reward={reward}")

            # check termination
            if done or episode_t == self.env.max_episode_steps:
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.agent.gamma * G

                # store the return
                self.trn_returns.append(G)
                episode_idx = len(self.trn_returns)

                # evaluate the policy
                if not np.mod(episode_idx + 1, self.trn_params['eval_policy_freq']):
                    self.eval()
                    # check if we should save the model
                    if self.eval_success[-1] >= last_eval_success:
                        # save the model
                        self.save_model(results_dir)
                        last_eval_success = self.eval_success[-1]

                # print the information
                pbar.set_description(
                    f"Ep={episode_idx} | "
                    f"G={np.mean(self.trn_returns[-10:]) if self.trn_returns else 0:.2f} | "
                    f"Eval={np.mean(self.eval_success[-10:]) if self.eval_success else 0:.2f} | "
                    f"Init={self.env.start_info['pos']} | "
                    f"Goal={self.env.goal_info['pos']}"
                )

                eval_res = np.mean(self.eval_success[-10:]) if self.eval_success else 0
                G_res = np.mean(self.trn_returns[-10:]) if self.trn_returns else 0

                # add plots to tensorboard
                self.tb.add_scalar("Mean success rate", eval_res, episode_idx)
                self.tb.add_scalar("Expected return", G_res, episode_idx)
                # add loss function
                self.tb.add_scalar("Loss", self.agent.train_loss[-1] if len(self.agent.train_loss) > 0 else 0, episode_idx)

                # reset the environment
                episode_t, rewards = 0, []
                obs = self.reset()
                start_pos = obs['observation']
                goal_pos = obs['desired_goal']
            else:
                # increment
                obs = next_obs
                episode_t += 1

            if t > self.start_train_step:
                # update the behavior model
                if not np.mod(t, self.update_policy_freq):
                    batch_data = self.memory.sample_batch(self.batch_size)
                    self.agent.update_behavior_policy(batch_data)

                # update the target model
                if not np.mod(t, self.update_target_freq):
                    self.agent.update_target_policy()

        # save the results
        self.save_data(results_dir)
        self.tb.close()


class HERDQNExperiment(object):
    def __init__(self, agent, env, test_env, trn_params):
        # initialize the experiment
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.trn_params = trn_params
        self.env_params = self._get_env_params()

        # training parameters
        self.use_obs = trn_params['use_obs']
        self.schedule = LinearSchedule(1, 0.01, trn_params['total_time_steps'] / 2)
        # if use her
        self.use_her = trn_params['use_her']
        # create the sampling function
        self.sampler_func = HER(replay_strategy='future',
                                replay_k=4,
                                reward_func=self.compute_reward)
        # define the memory buffer
        if not trn_params['use_obs']:
            self.memory = SampleFuncReplayBuffer(env_params=self.env_params,
                                                 size_in_transitions=trn_params['memory_size'],
                                                 sample_func=self.sampler_func.sample_transitions)
        else:
            self.memory = SampleFuncReplayBufferImages(env_params=self.env_params,
                                                       size_in_transitions=trn_params['memory_size'],
                                                       sample_func=self.sampler_func.sample_transitions_images)

        self.start_train_step = trn_params['start_train_step']
        self.total_time_steps = trn_params['total_time_steps']
        self.update_policy_freq = trn_params['update_policy_freq']
        self.update_target_freq = trn_params['update_target_freq']
        self.memory_size = trn_params['memory_size']
        self.batch_size = trn_params['batch_size']

        # special modifications
        self.use_her = trn_params['use_her']  # whether use her

        # save results
        self.trn_loss = []
        self.trn_returns = []
        self.eval_success = []
        self.save_dir = trn_params['save_dir']
        self.model_name = trn_params['model_name']

        current_time = datetime.now()
        date = current_time.strftime("%m-%d")
        time = current_time.strftime("%H-%M-%S")
        label = self.trn_params['run_label']
        target_log_dir = os.path.join('./runs',
                                      label,
                                      date,
                                      f"{time}_{trn_params['env_name'].split('.')[0]}_"
                                      f"label={self.trn_params['run_label']}_"
                                      f"{self.trn_params['env_name'].split('.')[0]}_"
                                      f"{self.trn_params['batch_size']}_"
                                      f"{self.trn_params['update_target_freq']}")

        # create the summary writer
        self.tb = SummaryWriter(log_dir=target_log_dir)
    # get the parameters of the environment
    def _get_env_params(self):
        obs = self.env.reset()
        action = self.env.action_space.sample()
        env_params = {
            'max_episode_step': self.env.max_episode_steps,
            'obs': obs['observation'],
            'ag': obs['achieved_goal'],
            'dg': np.array(obs['desired_goal']),
            'act': np.array([action])
        }
        return env_params

    def reset(self):
        return self.env.reset()

    def get_action(self, obs):
        return self.agent.get_action(obs)

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        return next_obs, reward, done, {}

    def eval(self):
        # number of the evaluation
        eval_total_num = 100
        success_num = 0.0
        optimal_steps = 10
        # start evaluation
        for i in range(eval_total_num):
            # reset the environment
            obs = self.reset()
            for t in range(optimal_steps):
                # get action
                self.agent.eps = 0
                action = self.get_action(obs)
                # step in the environment
                next_obs, reward, done, _ = self.step(action)

                if reward == 0:
                    success_num += 1
                    break
                else:
                    obs = next_obs

        # record the success
        self.eval_success.append(success_num / eval_total_num)

    def save(self):
        pass

    def save_model(self, save_dir):
        torch.save(self.agent.behavior_policy_net.state_dict(),
                   save_dir + '/model/' + self.trn_params['model_name'] + '.pt')

    def save_data(self, save_dir):
        np.save(f'./{save_dir}/trn_return.npy', self.trn_returns)
        np.save(f'./{save_dir}/trn_acc.npy', self.eval_success)

    def compute_reward(self, achieved_goal, desired_goal):
        # reward function to compute the reward for the sampled batch using HER
        if not self.use_obs:
            distances = np.sum((achieved_goal - desired_goal) ** 2, axis=1)
        else:
            distances = np.sum((achieved_goal - desired_goal) ** 2, axis=(1, 2, 3))

        # compute the termination
        dones = distances > 0
        if self.trn_params['sparse_reward'] == 0:
            rewards = 0 - dones
        else:
            rewards = (1 - dones) * self.trn_params['sparse_reward']
        return rewards

    def compute_terminal(self, batch_data):
        if self.trn_params['sparse_reward'] == 0:
            batch_data['done'] = (1 - np.abs(batch_data['reward'])).astype(np.bool)
        else:
            batch_data['done'] = (batch_data['reward'] / self.trn_params['sparse_reward']).astype(np.bool)

        return batch_data

    def run(self):
        # training variables
        episode_t = 0
        episode_idx = 0
        rewards = []

        # create the saving directory
        current_time = datetime.today()
        run_mode = os.path.join(self.trn_params['run_label'],
                                current_time.strftime("%m-%d"),
                                current_time.strftime("%H-%M-%S") +
                                f"_{self.trn_params['run_label']}_{self.trn_params['env_name'].split('.')[0]}")
        results_dir = os.path.join(self.trn_params['save_dir'], run_mode)

        # remove the old one
        if os.path.exists(results_dir):
            os.system(f"rm -r {results_dir}")
        # create the new one
        os.makedirs(f"{results_dir}/model")
        os.makedirs(f"{results_dir}/data")

        # reset the environment
        obs = self.env.reset()

        # position info for debug print
        start_pos = self.env.start_info['pos']
        goal_pos = self.env.goal_info['pos']

        # variables for rollout episodes
        ep_obs, ep_ag, ep_dg, ep_act = [], [], [], []
        ep_obs.append(obs['observation'])  # init observation
        ep_ag.append(obs['achieved_goal'])  # init achieved goal
        goal_obs = obs['desired_goal']
        # start training
        pbar = tqdm.trange(self.total_time_steps)
        last_eval_success = 0
        for t in pbar:
            # get one action
            self.agent.eps = self.schedule.get_value(t)
            action = self.get_action(obs)
            # step in the environment
            next_obs, reward, done, _ = self.step(action)

            # add to the buffer
            ep_obs.append(next_obs['observation'].copy())  # list of current observation T
            ep_ag.append(next_obs['achieved_goal'].copy())  # list of achieved observation T
            ep_dg.append(next_obs['desired_goal'].copy())  # list of goal observation T
            ep_act.append([action].copy())  # list of actions T - 1

            # reward
            rewards.append(reward)

            # check termination
            if not np.mod(episode_t + 1, self.env.max_episode_steps):
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.agent.gamma * G

                # store the return
                self.trn_returns.append(G)
                episode_idx = len(self.trn_returns)

                # evaluate the policy
                if not np.mod(episode_idx + 1, self.trn_params['eval_policy_freq']):
                    self.eval()
                    # check if we should save the model
                    if self.eval_success[-1] >= last_eval_success:
                        # save the model
                        self.save_model(results_dir)
                        last_eval_success = self.eval_success[-1]

                # print the information
                pbar.set_description(
                    f"Ep={episode_idx} | "
                    f"G={np.mean(self.trn_returns[-10:]) if self.trn_returns else 0:.2f} | "
                    f"Eval={np.mean(self.eval_success[-10:]) if self.eval_success else 0:.2f} | "
                    f"Init={start_pos} | "
                    f"Goal={goal_pos} | "
                    f"Memory size={len(self.memory)} | "
                )

                eval_res = np.mean(self.eval_success[-10:]) if self.eval_success else 0
                G_res = np.mean(self.trn_returns[-10:]) if self.trn_returns else 0
                self.tb.add_scalar("Expected return", G_res, episode_idx)
                self.tb.add_scalar("Mean success rate", eval_res, episode_idx)
                self.tb.add_scalar("TD loss", self.agent.train_loss[-1] if len(self.agent.train_loss) else 0, episode_idx)

                # store the episode into batch buffer
                mb_obs = np.array([ep_obs])
                mb_act = np.array([ep_act])
                mb_ag = np.array([ep_ag])
                mb_dg = np.array([ep_dg])
                self.memory.store_episode([mb_obs, mb_ag, mb_dg, mb_act])

                # reset the environment
                episode_t, rewards = 0, []
                ep_obs, ep_ag, ep_dg, ep_act = [], [], [], []
                obs = self.reset()
                ep_obs.append(obs['observation'].copy())
                ep_ag.append(obs['achieved_goal'].copy())
                goal_obs = obs['desired_goal']
                start_pos = self.env.start_info['pos']
                goal_pos = self.env.goal_info['pos']
            else:
                # increment
                obs = next_obs
                episode_t += 1

            if t > self.start_train_step:
                # update the behavior model
                if not np.mod(t, self.update_policy_freq):
                    batch_data = self.memory.sample(self.batch_size)
                    batch_data = self.compute_terminal(batch_data)
                    # re-arrange data list
                    batch_data_list = [batch_data['obs'],
                                       batch_data['action'],
                                       batch_data['reward'],
                                       batch_data['next_obs'],
                                       batch_data['done'],
                                       batch_data['goal']]
                    self.agent.update_behavior_policy(batch_data_list)

                # update the target model
                if not np.mod(t, self.update_target_freq):
                    self.agent.update_target_policy()

        # save the results
        self.save_data(results_dir)
        self.tb.close()
