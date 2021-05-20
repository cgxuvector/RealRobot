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
    parser.add_argument("--random_init", action="store_true", default=True)
    parser.add_argument("--random_goal", action="store_true", default=True)
    parser.add_argument("--reach_goal_eps", type=float, default=0.5)
    parser.add_argument("--max_episode_steps", type=int, default=10)
    parser.add_argument("--room_size", type=int, default=3)
    parser.add_argument("--sample_dist", type=int, default=1)

    # arguments for agent
    parser.add_argument("--dqn_mode", type=str, default="double")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sparse_reward", type=float, default=0)
    parser.add_argument("--model_path", type=str, default="./results/goal_dqn_double_rs_3_dist_1/05-18/"
                                                          "20-20-23_goal_dqn_double_rs_3_dist_1_test_env/"
                                                          "model/")

    return parser.parse_args()


def make_env(env_params):
    # create the maze from text
    maze = GoalTextMaze(text_file='env/mazes/maze_9_0.txt',
                        room_size=env_params['room_size'],
                        wall_size=env_params['wall_size'],
                        obs_name=env_params['obs_name'],
                        max_episode_steps=env_params['max_episode_step'],
                        rnd_init=env_params['random_init'],
                        rnd_goal=env_params['random_goal'],
                        dist=env_params['sample_dist'],
                        goal_reach_eps=env_params['goal_reach_eps'])
    return maze


def make_agent(agent_params, env_params):
    # create the agent
    agent = GoalDQNAgent(env_params, agent_params)
    agent.behavior_policy_net.load_state_dict(torch.load(agent_params['model_path'] + 'test_model.pt',
                                                         map_location=agent_params['device']))
    agent.behavior_policy_net.eval()
    agent.eps = 0

    return agent


def visualize_policy(agent, env):
    test_episode_num = 100
    max_episode_steps = 10
    success_count = 0
    for e in range(test_episode_num):
        obs = env.reset()
        env.render()
        print("+++++++++++++++++++++++++++++++")
        print(f"id {e}: start = {env.start_info['pos']}, goal = {env.goal_info['pos']}")
        for t in range(max_episode_steps):
            # get action
            agent.eps = 0
            action = agent.get_action(obs)
            # step
            next_obs, reward, done, _ = env.step(action)
            # render
            env.render()
            if reward == 0:
                print(f"start = {env.start_info['pos']}, goal = {env.goal_info['pos']} is True")
                print("+++++++++++++++++++++++++++++++")
                success_count += 1
                break
            else:
                obs = next_obs

    print(f"Success rate = {success_count / test_episode_num}")


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
        'sample_dist': args.sample_dist,
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
