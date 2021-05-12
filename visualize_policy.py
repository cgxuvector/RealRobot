from agent.GoalDQNAgent import GoalDQNAgent
from env.Maze import GoalTextMaze
import matplotlib.pyplot as plt

import IPython.terminal.debugger as Debug

import argparse
import torch


def parse_input_arguments():
    parser = argparse.ArgumentParser()

    # arguments for environment
    parser.add_argument("--observation", type=str, default="panorama-depth")
    parser.add_argument("--panorama_mode", type=str, default='concat')
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--random_goal", action="store_true", default=False)
    parser.add_argument("--reach_goal_eps", type=float, default=0.5)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--room_size", type=int, default=1)

    # arguments for agent
    parser.add_argument("--dqn_mode", type=str, default="double")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sparse_reward", type=float, default=0)
    parser.add_argument("--model_path", type=str, default="./results/double_dqn_depth/04-27/"
                                                          "17-14-10_double_dqn_depth_test_env/model/")

    return parser.parse_args()


def make_env(env_params):
    # create the maze from text
    maze = GoalTextMaze(text_file='env/mazes/maze_test.txt',
                        room_size=env_params['room_size'],
                        wall_size=env_params['wall_size'],
                        obs_name=env_params['obs_name'],
                        max_episode_steps=env_params['max_episode_step'],
                        rnd_init=env_params['random_init'],
                        rnd_goal=env_params['random_goal'],
                        goal_reach_eps=env_params['goal_reach_eps'])
    return maze


def make_agent(agent_params, env_params):
    # create the agent
    agent = GoalDQNAgent(env_params, agent_params)
    agent.behavior_policy_net.load_state_dict(torch.load(agent_params['model_path'] + 'double_dqn_depth_panorama.pt',
                                                         map_location=agent_params['device']))
    agent.behavior_policy_net.eval()
    agent.eps = 0

    return agent


def visualize_policy(agent, env):
    # reset
    obs = env.reset()
    env.render()

    for t in range(50):
        # get action
        action = agent.get_action(obs)
        # step
        next_obs, reward, done, _ = env.step(action)
        print(env.agent.pos)
        # render
        env.render()
        if reward == 0:
            break
        else:
            obs = next_obs


if __name__ == "__main__":
    # inputs arguments
    args = parse_input_arguments()

    # define environment parameters
    my_env_params = {
        'room_size': args.room_size,  # room size
        'wall_size': 0.01,  # size of the wall
        'obs_name': args.observation,
        'panorama_mode': args.panorama_mode,
        'max_episode_step': args.max_episode_steps,
        'random_init': args.random_init,
        'random_goal': args.random_goal,
        'goal_reach_eps': args.reach_goal_eps
    }

    # define agent parameters
    my_agent_params = {
        'dqn_mode': args.dqn_mode,
        'obs_name': args.observation,
        'device': args.device,
        'model_path': args.model_path,

        # other unnecessary parameters
        'use_obs': False,
        'gamma': 0.995,
        'lr': 1e-4
    }

    # make the environment
    my_env = make_env(my_env_params)

    # make the agent
    my_agent = make_agent(my_agent_params, my_env_params)

    # run visualization
    visualize_policy(my_agent, my_env)
