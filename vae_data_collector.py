from env.Maze_v1 import GoalTextMaze
import random
import numpy as np
import json

import IPython.terminal.debugger as Debug


def make_env(maze_size, maze_id):
    # create the maze from text
    maze = GoalTextMaze(text_file=f'env/mazes/maze_{maze_size}_{maze_id}.txt',
                        room_size=3.0,
                        wall_size=0.01,
                        obs_name='panorama-depth',
                        max_episode_steps=100,
                        action_space='3-actions',
                        rnd_init=True,
                        rnd_goal=True,
                        agent_rnd_spawn=True,
                        goal_rnd_spawn=True,
                        dist=1,
                        goal_reach_eps=0.05,
                        eval_mode=False,
                        view='train',
                        obs_width=80,
                        obs_height=60)

    return maze


def collect_data(env, size, idx, sample_per_room=20):
    # reset the environment
    env.reset()
    # valid positions on map
    valid_locations = env.valid_locations
    # loop all valid locations
    maze_data = []
    for loc in valid_locations:
        # crop the local map
        loc_map = env.array_map[loc[0]:loc[0]+3, loc[1]:loc[1]+3].astype(np.float32).tolist()
        # get the room
        room = env.data_rooms[loc[0]][loc[1]]
        # sample the location in the room
        # For each room, randomly sample 20 positions
        room_data = []
        for i in range(sample_per_room):
            x = random.uniform(room.min_x + 0.5, room.max_x - 0.5)
            z = random.uniform(room.min_z + 0.5, room.max_z - 0.5)
            # place the agent at the place
            env.place_entity(ent=env.agent, room=room, pos=np.array([x, 0, z]), dir=0)
            # render the observation
            obs = env._render_customize_obs()
            # print the info
            print(f"{i}: Place the agent at room {loc} with location {env.agent.pos}")
            # store the data
            room_data.append({'room id': [float(item) for item in loc],
                              'local map': loc_map,  # convert to list
                              'position observation': [x, z],
                              'depth observation': [o.tolist() for o in obs]})  # convert to list

        maze_data.append(room_data)

    # save the results
    with open(f'./data/vae/vae_data_maze_{size}_{idx}.json', 'w') as f_out:
        json.dump(maze_data, f_out)
    f_out.close()
    env.close()


if __name__ == "__main__":
    maze_size_list = [7, 9, 11]
    for m in maze_size_list:
        for i in range(20):
            # make the environment
            my_env = make_env(m, i)
            # create
            collect_data(my_env, m, i, 20)


