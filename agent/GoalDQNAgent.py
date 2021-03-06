import torch
import numpy as np
from torch import nn
from model.DeepNetwork import GoalDeepQNet

import matplotlib.pyplot as plt

import IPython.terminal.debugger as Debug


# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)


class GoalDQNAgent(object):
    # initialize the agent
    def __init__(self,
                 env_params=None,
                 agent_params=None,
                 ):
        # save the parameters
        self.env_params = env_params
        self.agent_params = agent_params

        # environment parameters
        if env_params['action_space'] == '4-actions':
            self.action_space = np.linspace(0, 4, 4, endpoint=False).astype('uint8')
            self.action_dim = 4
        elif env_params['action_space'] == '3-actions':
            self.action_space = np.linspace(0, 3, 3, endpoint=False).astype('uint8')
            self.action_dim = 3
        else:
            raise Exception("Invalid action space.")

        # create behavior policy and target networks
        self.dqn_mode = agent_params['dqn_mode']
        self.use_obs = agent_params['use_obs']
        self.gamma = agent_params['gamma']
        self.behavior_policy_net = GoalDeepQNet(env_params['obs_name'],
                                                env_params['panorama_mode'],
                                                act_dim=self.action_dim)
        self.target_policy_net = GoalDeepQNet(env_params['obs_name'],
                                              env_params['panorama_mode'],
                                              act_dim=self.action_dim)

        # initialize target network with behavior network
        self.behavior_policy_net.apply(customized_weights_init)
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

        # send the agent to a specific device: cpu or gpu
        self.device = torch.device(agent_params['device'])
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=self.agent_params['lr'])

        # other parameters
        self.eps = 1
        self.train_loss = []

    # get action
    def get_action(self, obs):
        if np.random.random() < self.eps:  # with probability eps, the agent selects a random action
            action = np.random.choice(self.action_space, 1)[0]
        else:  # with probability 1 - eps, the agent selects a greedy policy
            current_obs, goal_obs = self._obs_to_tensor(obs)
            with torch.no_grad():
                q_values = self.behavior_policy_net(current_obs, goal_obs)
                action = q_values.max(dim=1)[1].item()
        return action

    # update behavior policy
    def update_behavior_policy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']
        goal_tensor = batch_data_tensor['goal']

        # compute the q value estimation using the behavior network
        pred_q_value = self.behavior_policy_net(obs_tensor, goal_tensor)
        pred_q_value = pred_q_value.gather(dim=1, index=actions_tensor)

        # compute the TD target using the target network
        if self.dqn_mode == 'vanilla':
            # compute the TD target using vanilla method: TD = r + gamma * max a' Q(s', a')
            # no gradient should be tracked
            with torch.no_grad():
                max_next_q_value = self.target_policy_net(next_obs_tensor, goal_tensor).max(dim=1)[0].view(-1, 1)
                td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
        else:
            # compute the TD target using double method: TD = r + gamma * Q(s', argmaxQ_b(s'))
            with torch.no_grad():
                max_next_actions = self.behavior_policy_net(next_obs_tensor, goal_tensor).max(dim=1)[1].view(-1, 1).long()
                max_next_q_value = self.target_policy_net(next_obs_tensor, goal_tensor).gather(dim=1, index=max_next_actions).view(
                    -1, 1)
                td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
                td_target_value = td_target_value.detach()

        # compute the loss
        td_loss = torch.nn.functional.mse_loss(pred_q_value, td_target_value)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        self.train_loss.append(td_loss.item())

    # update update target policy
    def update_target_policy(self):
        tau = self.agent_params['polyak']
        if tau == 0:
            self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())
        else:
            assert tau >= 0.9, f"Invalid tau value, expected to be > 0.9, but get{tau}"
            for param, target_param in zip(self.behavior_policy_net.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)

    # load trained model
    def load_model(self, model_file):
        # load the trained model
        self.behavior_policy_net.load_state_dict(torch.load(model_file, map_location=self.device))
        self.behavior_policy_net.eval()

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _obs_to_tensor(self, obs):
        if self.env_params['obs_name'] == "state":
            obs_s = self._arr_to_tensor(np.array(obs['observation'])).view(1, -1)
            obs_g = self._arr_to_tensor(np.array(obs['desired_goal'])).view(1, -1)
        elif self.env_params['obs_name'] == 'rgb' or self.env_params['obs_name'] == 'depth':
            obs_s = torch.tensor(obs['observation'], device=self.device).float().unsqueeze(dim=0).permute(0, 3, 1, 2)
            obs_g = torch.tensor(obs['desired_goal'], device=self.device).float().unsqueeze(dim=0).permute(0, 3, 1, 2)
        elif self.env_params['obs_name'] == 'panorama-rgb' or self.env_params['obs_name'] == 'panorama-depth':
            obs_s = torch.tensor(np.array(obs['observation']), device=self.device).float().permute(0, 3, 1, 2)
            obs_g = torch.tensor(np.array(obs['desired_goal']), device=self.device).float().permute(0, 3, 1, 2)
        else:
            raise Exception('Invalid observation name')

        return obs_s, obs_g

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': [], 'goal': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr, goal_arr = batch_data

        if self.env_params['obs_name'] == 'state':
            # convert to tensors
            batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
            batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
            batch_data_tensor['goal'] = torch.tensor(goal_arr, dtype=torch.float32).to(self.device)
        elif self.env_params['obs_name'] == 'rgb' or self.env_params['obs_name'] == 'depth':
            batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
            batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
            batch_data_tensor['goal'] = torch.tensor(goal_arr, dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
        elif self.env_params['obs_name'] == 'panorama-rgb':
            batch_data_tensor['obs'] = torch.tensor(np.array(obs_arr), dtype=torch.float32).to(self.device).reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
            batch_data_tensor['next_obs'] = torch.tensor(np.array(next_obs_arr), dtype=torch.float32).to(self.device).reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
            batch_data_tensor['goal'] = torch.tensor(np.array(goal_arr), dtype=torch.float32).to(self.device).reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
        elif self.env_params['obs_name'] == 'panorama-depth':
            batch_data_tensor['obs'] = torch.tensor(np.array(obs_arr), dtype=torch.float32).to(self.device).reshape(-1, 32, 32, 1).permute(0, 3, 1, 2)
            batch_data_tensor['next_obs'] = torch.tensor(np.array(next_obs_arr), dtype=torch.float32).to(self.device).reshape(-1, 32, 32, 1).permute(0, 3, 1, 2)
            batch_data_tensor['goal'] = torch.tensor(np.array(goal_arr), dtype=torch.float32).to(self.device).reshape(-1, 32, 32, 1).permute(0, 3, 1, 2)
        else:
            raise Exception("Error")

        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        return batch_data_tensor
