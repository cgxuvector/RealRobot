from experiment.GoalDQNExperiment import GoalDQNExperiment
from agent.GoalDQNAgent import GoalDQNAgent

from env.Maze import TextMaze

import IPython.terminal.debugger as Debug


env_params = {
    # for making agent
    'act_num': 4,
    'obs_dim': (32, 32, 3),
    'goal_dim': (32, 32, 3),

    # for making environment
    'obs_name': 'rgb',
    'random_init': False,
    'random_goal': False,
}

agent_params = {
    'dqn_mode': "vanilla",
    'use_obs': False,
    'gamma': 0.95,
    'device': "cuda:0",
    'lr': 1e-4,
}

training_params = {
    # for environment
    'env_name': "maze_7",
    # for observation
    'use_obs': True,
    # for training details
    'start_train_step': 1000,
    'total_time_steps': 100000,
    'memory_size': 50000,
    'batch_size': 32,
    'update_policy_freq': 4,
    'update_target_freq': 2000,
    # for logging
    'save_dir': './results',
    'model_name': 'test_model',
    'run_label': 'test_run'
}


def make_agent():
    agent = GoalDQNAgent(env_params, agent_params)
    return agent


def make_envs():
    # create the training environment
    trn_env = TextMaze(text_file='./env/maze_test.txt',
                       room_size=2,
                       wall_size=0.01,
                       max_episode_steps=0.5,
                       obs_name=env_params['obs_name'],
                       rnd_init=env_params['random_init'],
                       rnd_goal=env_params['random_goal'])

    # creat the testing environment
    tst_env = TextMaze(text_file='./env/maze_test.txt',
                       room_size=2,
                       wall_size=0.01,
                       max_episode_steps=0.5,
                       obs_name=env_params['obs_name'],
                       rnd_init=env_params['random_init'],
                       rnd_goal=env_params['random_goal'])

    return [trn_env, tst_env]


if __name__ == "__main__":
    # make the dqn agent
    myAgent = make_agent()

    # make the training and testing environments
    envs = make_envs()

    # make the experiment runner
    myExperiment = GoalDQNExperiment(myAgent, envs[0], envs[1], training_params)
    myExperiment.run()
