from env.Maze import GoalTextMaze

import matplotlib.pyplot as plt

import IPython.terminal.debugger as Debug

if __name__ == "__main__":
    # test code
    maze_configurations = {
        'room_size': 3,  # room size
        'wall_size': 0.01,  # size of the wall
        'obs_name': 'panorama-rgb',
        'max_episode_step': 1000,
        'random_init': False,
        'random_goal': False,
        'action_num': 3
    }

    # create the maze from text
    myMaze = GoalTextMaze(text_file='env/mazes/maze_test.txt',
                          room_size=maze_configurations['room_size'],
                          wall_size=maze_configurations['wall_size'],
                          obs_name=maze_configurations['obs_name'],
                          max_episode_steps=maze_configurations['max_episode_step'],
                          rnd_init=maze_configurations['random_init'],
                          rnd_goal=maze_configurations['random_goal'],
                          action_num=maze_configurations['action_num'])

    # reset
    obs = myMaze.reset()
    myMaze.render()

    print(f"Start pos = {myMaze.start_info['pos']}, goal pos = {myMaze.goal_info['pos']}")

    # start test
    for i in range(50):
        # random sample an action from the action space
        act = myMaze.action_space.sample()

        next_obs, reward, done, _ = myMaze.step(act)

        print(myMaze.agent.pos, " - ", myMaze.agent.dir)

        obs = next_obs

        myMaze.render()
        if done:
            myMaze.reset()

