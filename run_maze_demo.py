from env.Maze import GoalTextMaze

import matplotlib.pyplot as plt

import IPython.terminal.debugger as Debug

if __name__ == "__main__":
    # test code
    maze_configurations = {
        'room_size': 3,  # room size
        'wall_size': 0.01,  # size of the wall
        'obs_name': 'state',
        'max_episode_step': 1000,
        'random_init': False,
        'random_goal': False
    }

    # create the maze from text
    myMaze = GoalTextMaze(text_file='env/mazes/maze_test.txt',
                          room_size=maze_configurations['room_size'],
                          wall_size=maze_configurations['wall_size'],
                          obs_name=maze_configurations['obs_name'],
                          max_episode_steps=maze_configurations['max_episode_step'],
                          rnd_init=maze_configurations['random_init'],
                          rnd_goal=maze_configurations['random_goal'])

    # reset
    obs = myMaze.reset()
    myMaze.render()

    print(f"Start pos = {myMaze.start_info['pos']}, goal pos = {myMaze.goal_info['pos']}")

    # start test
    for i in range(50):
        # random sample an action from the action space
        act = myMaze.action_space.sample()
        act = 0
        # if i == 0:
        #     act = 1
        # else:
        #     act = 2
        # step
        next_obs, reward, done, _ = myMaze.step(act)

        print(f"Current state = {obs['observation']},"
              f" action = {myMaze.ACTION_NAME[act]},"
              f" next state = {next_obs['observation']}")

        # print(myMaze.agent.pos)

        obs = next_obs

        myMaze.render()
        if done:
            myMaze.reset()

