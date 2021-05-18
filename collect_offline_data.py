import random
import json
import argparse
import os
from env.Maze import GoalTextMaze
import numpy as np

import IPython.terminal.debugger as Debug


# define the A star expert policy
class RandomPolicy(object):
    """
        Collect the optimal episodes from multiple environment
    """

    def __init__(self, env_list, configs):
        super(RandomPolicy, self).__init__()
        self.env_list = env_list
        self.sample_episode_num = configs.sample_num
        self.rnd_seed = configs.rnd_seed
        self.results_data = []
        self.configs = configs

    def sample_episodes(self, configs):
        # sample multiple episodes
        episodes = []
        action_list = list(range(self.configs.action_num))
        action_name = ['turn_left', 'turn_right', 'forward', 'backward']

        # create the maze and the map
        env_maze = configs['maze']

        # start collecting episodes
        for i in range(configs['episode_num']):
            # sample the start and goal locations
            obs = env_maze.reset()
            # get the optimal trajectory using random policy

            # compose the obs
            obs = {'o': [o.tolist() for o in obs['observation']],
                   's': [env_maze.agent.pos[0],
                         env_maze.agent.pos[2],
                         np.sin(env_maze.agent.dir),
                         np.cos(env_maze.agent.dir)]}

            episode = [obs]
            # start collecting one episode
            for t in range(configs['maze'].max_episode_steps):
                # sample one action
                action = random.sample(action_list, 1)[0]
                # step
                next_obs, reward, done, _ = env_maze.step(action)
                # store the trajectory
                episode.append(action_name[action_list.index(action)])

                # compose the next obs
                next_obs = {'o': [o.tolist() for o in next_obs['observation']],
                            's': [env_maze.agent.pos[0],
                                  env_maze.agent.pos[2],
                                  np.sin(env_maze.agent.dir),
                                  np.cos(env_maze.agent.dir)]}
                episode.append(next_obs)

            # save the episode
            print(f"Process maze id {configs['id']}: {i + 1} episode.")
            episodes.append(episode)
        env_maze.close()
        # put the data to the queue
        return {'id': configs['id'], 'map': configs['map'].tolist(), 'episodes': episodes}

    def run(self):
        # construct the configurations for each maze
        for idx in range(len(self.env_list)):
            configs = {'id': idx,
                       'maze': self.env_list[idx][0],
                       'map': self.env_list[idx][1],
                       'episode_num': self.configs.sample_num
                       }

            # run mazes one by one
            res = self.sample_episodes(configs)
            self.results_data.append(res)

        # save the results
        self.save_results()

    # save function
    def save_results(self):
        with open(f'./data/maze_{self.configs.maze_size}x{self.configs.maze_size}_0-{self.configs.maze_num - 1}'
                  f'_act{self.configs.action_num}_{self.configs.obs_name}_rnd.json',
                  "w") as f_out:
            json.dump(self.results_data, f_out)
        f_out.close()


def load_maze(m_id, configs):
    maze_file = configs.maze_path + f"maze_{configs.maze_size}_{m_id}.txt"
    # load the map
    with open(maze_file, 'r') as f_in:
        lines = f_in.readlines()
    f_in.close()

    maze_map = np.array([[int(float(d)) for d in l.rstrip().split(',')] for l in lines])
    maze_map = np.where(maze_map == 2, 1, maze_map)

    # set up the 3-D maze configs
    maze_configurations = {
        'room_size': configs.room_size,  # room size
        'wall_size': 0.01,  # size of the wall
        'obs_name': configs.obs_name,
        'max_episode_step': configs.max_episode_steps,
        'random_init': True,
        'random_goal': True,
        'action_num': configs.action_num
    }

    # create the maze from text
    maze = GoalTextMaze(text_file=maze_file,
                        room_size=maze_configurations['room_size'],
                        wall_size=maze_configurations['wall_size'],
                        obs_name=maze_configurations['obs_name'],
                        max_episode_steps=maze_configurations['max_episode_step'],
                        rnd_init=maze_configurations['random_init'],
                        rnd_goal=maze_configurations['random_goal'],
                        action_num=maze_configurations['action_num'])
    return [maze, maze_map]


def parse_input():
    parser = argparse.ArgumentParser()
    # data load and save path
    parser.add_argument("--maze_path", type=str, default="./env/mazes/")
    parser.add_argument("--save_path", type=str, default="./data")
    # collection parameters
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--maze_num", type=int, default=20)
    parser.add_argument("--policy_type", type=str, default="random")
    parser.add_argument("--rnd_seed", type=int, default=4213)
    # maze parameters
    parser.add_argument("--maze_size", type=int, default=7)
    parser.add_argument("--room_size", type=int, default=3)
    parser.add_argument("--obs_name", type=str, default="state")
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--action_num", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    # parse the input arguments
    args = parse_input()

    # store the maze list
    maze_list = []
    for i in range(args.maze_num):
        maze_env = load_maze(i, args)
        maze_list.append(maze_env)

    # create the random collector
    myCollector = RandomPolicy(maze_list, args)
    myCollector.run()


