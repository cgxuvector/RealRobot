from env.Maze_debug import GoalTextMaze
import random

import matplotlib.pyplot as plt

import IPython.terminal.debugger as Debug

# todo: the window should be changed to 60x60. however, for real robot, change back to 60x80

if __name__ == "__main__":
    # test code
    maze_configurations = {
        'room_size': 3,  # room size
        'wall_size': 0.01,  # size of the wall
        'obs_name': 'panorama-depth',
        'max_episode_step': 100,
        'random_init': False,
        'random_goal': False,
        'action_num': 4,
        'dist': 0,
        'rnd_spawn': False,
    }

    random.seed(4312)

    # create the maze from text
    myMaze = GoalTextMaze(text_file='env/mazes/maze_7_0.txt',
                          room_size=maze_configurations['room_size'],
                          wall_size=maze_configurations['wall_size'],
                          obs_name=maze_configurations['obs_name'],
                          max_episode_steps=maze_configurations['max_episode_step'],
                          rnd_init=maze_configurations['random_init'],
                          rnd_goal=maze_configurations['random_goal'],
                          dist=maze_configurations['dist'],
                          rnd_spawn=maze_configurations['rnd_spawn'],
                          action_num=maze_configurations['action_num'])

    # reset
    obs = myMaze.reset()
    myMaze.plot_panorama_point_cloud()
    myMaze.render()
    print(f"Start pos = {myMaze.start_info['pos']}, goal pos = {myMaze.goal_info['pos']}")
    # start test
    for i in range(100):
        # random sample an action from the action space
        act = myMaze.action_space.sample()
        act = 2
        next_obs, reward, done, _ = myMaze.step(act)
        obs = next_obs
        myMaze.plot_panorama_point_cloud()
        myMaze.render()
        Debug.set_trace()
        if done:
            myMaze.reset()

