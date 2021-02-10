import numpy as np
import math
import random
from gym import spaces
from gym_miniworld.miniworld import MiniWorldEnv, Room
from gym_miniworld.entity import Box, ImageFrame, Entity
from gym_miniworld.params import DEFAULT_PARAMS

import IPython.terminal.debugger as Debug


class TextMaze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=4,
        num_cols=8,
        room_size=3,
        wall_size=0.25,
        max_episode_steps=None,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = wall_size

        """ customized code """
        # start
        self.array_map, self.valid_locations = self._load_txt()
        # reset the rows and cols
        assert self.array_map.shape[0] > 2 and self.array_map.shape[1] > 2, \
            print(f"The maze size should be bigger than 2 x 2")
        self.num_rows = self.array_map.shape[0] - 2
        self.num_cols = self.array_map.shape[1] - 2
        # end

        super().__init__(
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def step(self, action, forward_step_size=0.5, turn_step_size=90):
        """
        Step function with customized forward/backward step size; turning step size
        :param action: action index (discrete version)
        :param forward_step_size: minimal forward step size (float smaller than 1 for bug free)
        :param turn_step_size: minimal turn step size (degree integer)
        :return: standard return like gym
        """
        info = {}

        self.step_count += 1

        fwd_step = forward_step_size
        fwd_drift = 0.0
        turn_step = turn_step_size

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.move_agent(-fwd_step, fwd_drift)

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)

        # Generate the current camera image
        obs = self.render_obs()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        return obs, reward, done, info

    """ Auxiliary functions
    """
    def _place_agent(self, room=None, pos=None, ori=0):
        return self.place_entity(
                    self.agent,
                    room=room,
                    pos=pos,
                    dir=ori
                )

    @staticmethod
    def _load_txt():
        # read the map from text file
        with open('./env/maze_test.txt', 'r') as f_in:
            file_data = f_in.readlines()
        f_in.close()
        map_data = [l.rstrip() for l in file_data]

        # get the rows and cols
        rows = len(map_data)
        cols = len(map_data[0])
        print(f"Maze row num = {rows}, col num = {cols}")
        # create the array map
        map_array = np.zeros((rows, cols))
        # store valid locations
        map_valid_locations = []
        # construct the map
        for l_idx, l in enumerate(map_data):
            for s_idx, s in enumerate(l):
                if s == "*":
                    # wall cell will be 1 and corridor cell will be 0
                    map_array[l_idx, s_idx] = 1
                    continue
                else:
                    map_valid_locations.append((l_idx - 1, s_idx - 1))

        return map_array, map_valid_locations

    def _gen_world(self):
        # customize this function to generate all the rooms
        rows = []

        # For each row
        locs = []
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                # check if the current position is valid
                if (j, i) not in self.valid_locations:
                    row.append(None)
                    continue

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall'
                )
                locs.append((j, i))
                row.append(room)

            rows.append(row)

        # record visited room
        visited = set()

        # connect the neighbors based on the map info
        for room_loc in self.valid_locations:
            # locate the current room
            room = rows[room_loc[0]][room_loc[1]]

            # add the room
            visited.add(room)

            # compute the neighbors
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, left, right

            # loop valid neighbors
            for di, dj in neighbors:
                ni = room_loc[1] + di
                nj = room_loc[0] + dj

                # check validation
                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue
                if (nj, ni) not in self.valid_locations:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

        # place the target
        self.box = self.place_entity(Box(color='red'))

        # place the agent randomly
        agent_room_loc = random.sample(self.valid_locations, 1)[0]
        print(f"Agent is spawned at room {agent_room_loc}")
        agent_room = rows[agent_room_loc[0]][agent_room_loc[1]]
        self._place_agent(room=agent_room)