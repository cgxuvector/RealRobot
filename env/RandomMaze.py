from gym import spaces
from env.BaseMaze import TextMaze
import numpy as np
import IPython.terminal.debugger as Debug
import matplotlib.pyplot as plt


class RandomMaze(object):
    def __init__(self, maze_configs):
        # maze environment
        self._maze = TextMaze(num_rows=maze_configs['num_rows'],
                              num_cols=maze_configs['num_cols'],
                              room_size=maze_configs["room_size"],
                              wall_size=maze_configs['wall_size'])

        # action space
        self.actions = ['turn_left', 'turn_right', 'forward', 'backward']
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
            obs = self._maze.render_depth()
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
            obs = self._maze.render_depth()
        elif self.observation_name == 'rgb-d':  # return RGB + depth observation; Shape: H x W x 4
            rgb_obs = obs
            d_obs = self._maze.render_depth()
            obs = np.concatenate((rgb_obs, d_obs), axis=2)
        elif self.observation_name == 'rgb':  # return RGB observation; Shape: H x W x 3
            obs = obs
        else:
            raise Exception("Invalid observation name.")

        plt.imshow(obs)
        plt.show()
        plt.pause(0.1)

        return obs

    def seed(self, seed=None):
        self._maze.seed(seed)

    def render(self):
        self._maze.render(view=self.configs['view_mode'])

    def compute_reward(self):
        pass