import matplotlib.pyplot as plt
import gym_miniworld
from gym_miniworld.envs import Maze


class RandomMaze(Maze):
    def __init__(self, maze_configs):
        super(RandomMaze, self).__init__()
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def render(self, mode='human', close=False, view='agent'):
        pass

    def compute_reward(self):
        pass
