from env.Maze_v1 import GoalTextMaze
import random

import matplotlib.pyplot as plt

import IPython.terminal.debugger as Debug

# todo: the window should be changed to 60x60. however, for real robot, change back to 60x80

if __name__ == "__main__":
    # test code
    maze_configurations = {
        'room_size': 0.78,  # room size
        'wall_size': 0.01,  # size of the wall
        'obs_name': 'panorama-depth',
        'max_episode_step': 100,
        'random_init': False,
        'random_goal': False,
        'action_num': 4,
        'dist': 0,
        'agent_rnd_spawn': False,
        'goal_rnd_spawn': False,
        'action_space': '3-actions',
        'view': 'all',
        'obs_width': 160,
        'obs_height': 120,
        'agent_radius': 0.175,
        'forward_step_size': 0.1,

        'clip_depth_obs': True,
        'clip_max_val': 0.78 * 3 / 2
    }

    random.seed(4312)

    # create the maze from text
    myMaze = GoalTextMaze(text_file='env/mazes/maze_7_0.txt',
                          room_size=maze_configurations['room_size'],
                          forward_step_size=maze_configurations['forward_step_size'],
                          wall_size=maze_configurations['wall_size'],
                          obs_name=maze_configurations['obs_name'],
                          max_episode_steps=maze_configurations['max_episode_step'],
                          rnd_init=maze_configurations['random_init'],
                          rnd_goal=maze_configurations['random_goal'],
                          dist=maze_configurations['dist'],
                          agent_rnd_spawn=maze_configurations['agent_rnd_spawn'],
                          goal_rnd_spawn=maze_configurations['goal_rnd_spawn'],
                          action_space=maze_configurations['action_space'],
                          view=maze_configurations['view'],
                          agent_radius=maze_configurations['agent_radius'],
                          obs_width=maze_configurations['obs_width'],
                          obs_height=maze_configurations['obs_height'],
                          clip_depth_obs=maze_configurations['clip_depth_obs'],
                          clip_depth_max=maze_configurations['clip_max_val'])

    # reset
    obs = myMaze.reset()
    myMaze.render()
    print(f"Start pos = {myMaze.start_info['pos']}, goal pos = {myMaze.goal_info['pos']}")
    # start test
    for i in range(100):
        # random sample an action from the action space
        act = myMaze.action_space.sample()
        act = 2
        next_obs, reward, done, _ = myMaze.step(act)
        obs = next_obs
        myMaze.render()
        Debug.set_trace()
        if done:
            myMaze.reset()

