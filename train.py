from experiment.GoalDQNExperiment import GoalDQNExperiment
from agent.GoalDQNAgent import GoalDQNAgent

from env.Maze import GoalTextMaze

import argparse

import IPython.terminal.debugger as Debug


def parse_input_arguments():
    parser = argparse.ArgumentParser()

    # arguments for environment
    parser.add_argument("--observation", type=str, default="state")
    parser.add_argument("--panorama_mode", type=str, default='concat')
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--random_goal", action="store_true", default=False)
    parser.add_argument("--reach_goal_eps", type=float, default=0.5)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--room_size", type=int, default=1)

    # arguments for agent
    parser.add_argument("--dqn_mode", type=str, default="double")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)

    # arguments for training
    parser.add_argument("--env_name", type=str, default="test_env")
    parser.add_argument("--use_obs", action="store_true", default=False)
    parser.add_argument("--start_train_step", type=int, default=1000)
    parser.add_argument("--total_time_steps", type=int, default=2000000)
    parser.add_argument("--memory_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--update_policy_freq", type=int, default=4)
    parser.add_argument("--update_target_freq", type=int, default=2000)
    parser.add_argument("--eval_policy_freq", type=int, default=1000)

    # arguments for techniques
    parser.add_argument("--use_her", action="store_true", default=False)
    parser.add_argument("--sparse_reward", type=float, default=0)

    # for logging
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--model_name", type=str, default="test_model")
    parser.add_argument("--run_label", type=str, default="test_run")

    return parser.parse_args()


# parse the arguments
args = parse_input_arguments()

env_params = {
    # for making environment
    'obs_name': args.observation,
    'panorama_mode': args.panorama_mode,
    'random_init': args.random_init,
    'random_goal': args.random_goal,
    'goal_reach_eps': args.reach_goal_eps,
    'max_episode_steps': args.max_episode_steps,
    'room_size': args.room_size
}

agent_params = {
    'dqn_mode': args.dqn_mode,
    'use_obs': args.use_obs,
    'gamma': args.gamma,
    'device': args.device,
    'lr': args.lr,
}

training_params = {
    # for environment
    'env_name': args.env_name,
    # for observation
    'use_obs': args.use_obs,
    # for training details
    'start_train_step': args.start_train_step,
    'total_time_steps': args.total_time_steps,
    'memory_size': args.memory_size,
    'batch_size': args.batch_size,
    'update_policy_freq': args.update_policy_freq,
    'update_target_freq': args.update_target_freq,
    'eval_policy_freq': args.eval_policy_freq,
    # for logging
    'save_dir': args.save_dir,
    'model_name': args.model_name,
    'run_label': args.run_label,
    # for other techniques
    'use_her': args.use_her,
    # sparse reward
    'sparse_reward': args.sparse_reward
}


def make_agent():
    agent = GoalDQNAgent(env_params, agent_params)
    return agent


def make_envs():
    # create the training environment
    trn_env = GoalTextMaze(text_file='env/mazes/maze_test.txt',
                           room_size=env_params['room_size'],  # room size
                           wall_size=0.01,
                           max_episode_steps=env_params['max_episode_steps'],  # step size
                           obs_name=env_params['obs_name'],
                           rnd_init=env_params['random_init'],
                           rnd_goal=env_params['random_goal'],
                           goal_reach_eps=env_params['goal_reach_eps'])  # stop epsilon

    # creat the testing environment
    tst_env = GoalTextMaze(text_file='env/mazes/maze_test.txt',
                           room_size=env_params['room_size'],
                           wall_size=0.01,
                           max_episode_steps=env_params['max_episode_steps'],
                           obs_name=env_params['obs_name'],
                           rnd_init=env_params['random_init'],
                           rnd_goal=env_params['random_goal'],
                           goal_reach_eps=env_params['goal_reach_eps'])

    return [trn_env, tst_env]


if __name__ == "__main__":
    # make the dqn agent
    myAgent = make_agent()

    # make the training and testing environments
    envs = make_envs()

    # make the experiment runner
    if training_params['use_her']:
        myExperiment = HERDQNExperiment(myAgent, envs[0], envs[1], training_params)
    else:
        myExperiment = GoalDQNExperiment(myAgent, envs[0], envs[1], training_params)
    myExperiment.run()
