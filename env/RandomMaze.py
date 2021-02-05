import matplotlib.pyplot as plt
from enum import IntEnum
import gym_miniworld
from gym import spaces
from env.BaseMaze import TextMaze

import numpy as np


class RandomMaze(object):
    def __init__(self, maze_configs):
        # maze environment
        self._maze = TextMaze(num_rows=maze_configs['num_rows'],
                              num_cols=maze_configs['num_cols'],
                              room_size=maze_configs["room_size"],
                              wall_size=maze_configs['wall_size'])

        # action space
        self.actions = ['forward', 'turn_left', 'turn_right', 'stop']
        self.action_space = spaces.Discrete(len(self.actions))

        # observation space
        # note: we provide with three types of image-based observations
        # - rgb for RGB images
        # - rgb-d for RGBD images
        # - depth for depth images
        self.observation_name = maze_configs['obs_name']
        # this is unclear whether it is correct
        self.observation_space = TextMaze.observation_space

        # other configuration parameters
        self.configs = maze_configs

    def step(self, action):
        # step in the environment
        obs, reward, done, info = self._maze.step(action)

        # re-construct the observations
        if self.observation_name == "depth":  # return depth observation; Shape: H x W
            obs = self._maze.render_depth().squeeze(axis=2)
        elif self.observation_name == 'rgb-d':  # return RGB + depth observation; Shape: H x W x 4
            rgb_obs = obs
            d_obs = self._maze.render_depth()
            obs = np.concatenate((rgb_obs, d_obs), axis=2)
        elif self.observation_name == 'rgb':  # return RGB observation; Shape: H x W x 3
            obs = obs
        else:
            raise Exception("Invalid observation name.")

        return obs, reward, done, info

    def reset(self):
        # set the random
        self._maze.seed()
        # reset the environment
        obs = self._maze.reset()

        # re-construct the observations
        if self.observation_name == "depth":  # return depth observation; Shape: H x W
            obs = self._maze.render_depth().squeeze(axis=2)
        elif self.observation_name == 'rgb-d':  # return RGB + depth observation; Shape: H x W x 4
            rgb_obs = obs
            d_obs = self._maze.render_depth()
            obs = np.concatenate((rgb_obs, d_obs), axis=2)
        elif self.observation_name == 'rgb':  # return RGB observation; Shape: H x W x 3
            obs = obs
        else:
            raise Exception("Invalid observation name.")

        return obs

    def seed(self, seed=None):
        self._maze.seed(seed)

    def render(self):
        self._maze.render(view=self.configs['view_mode'])

    def compute_reward(self):
        pass


# test code
maze_configurations = {
    'num_rows': 5,  # number of the row rooms
    'num_cols': 5,  # number of the col rooms
    'room_size': 2,  # room size
    'wall_size': 0.5,  # size of the wall
    'obs_name': 'depth',
    'view_mode': 'top'
}

myMaze = RandomMaze(maze_configurations)
episode_len = 100
myMaze.reset()
for i in range(episode_len):
    act = myMaze.action_space.sample()
    myMaze.step(act)
    plt.pause(0.2)
    myMaze.render()