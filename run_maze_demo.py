from env.Maze import TextMaze

if __name__ == "__main__":
    # test code
    maze_configurations = {
        'room_size': 2,  # room size
        'wall_size': 0.01,  # size of the wall
        'obs_name': 'panorama-depth',
        'max_episode_step': 1000,
        'random_init': True,
        'random_goal': True
    }

    # create the maze from text
    myMaze = TextMaze(text_file='./env/maze_test.txt',
                      room_size=maze_configurations['room_size'],
                      wall_size=maze_configurations['wall_size'],
                      obs_name=maze_configurations['obs_name'],
                      max_episode_steps=maze_configurations['max_episode_step'],
                      rnd_init=maze_configurations['random_init'],
                      rnd_goal=maze_configurations['random_goal'])

    # reset
    obs = myMaze.reset()
    myMaze.render()

    # start test
    for i in range(1000):
        # random sample an action from the action space
        act = myMaze.action_space.sample()
        # step
        obs, reward, done, _ = myMaze.step(act)
        # render
        myMaze.render()

        if done:
            myMaze.reset()

