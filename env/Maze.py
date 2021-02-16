"""
    Maze construction cookbook:
        - Maze is initialized by generating N x M rooms with four walls surrounded.
        - Using the recursive backtracking algorithm, rooms are connected by removing walls between
          adjacent rooms.
        - Wall has thickness that will also affect the global shape of the constructed maze.

    In this version:
        E.g. To build a 5 x 5 maze, we follow the definition in DeepMind Lab where '*' represents the wall
             cell and ' ' represents the empty corridor cell. To remove the influence from the wall thickness,
             we choose to set the wall parameter to be 0.01.
"""
import numpy as np
import random
import copy as cp
from gym import spaces
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.entity import Box, Agent

import IPython.terminal.debugger as Debug


""" To do
        - compute reward
        - panorama view
"""


class TextMaze(MiniWorldEnv):
    """
    3-D Maze environment
        This 3-D environment is constructed based on maze.py in miniworld.
        We make the following modifications":
            - Initialize the maze from text files
            - Egocentric discrete action space
                - Forward, backward, turn left, and turn right.
                - Customizable action steps.
            - Customizable observations
                - Front view: RGB images
                - Front view: Depth images
    """
    def __init__(
        self,
        num_rows=8,
        num_cols=8,
        room_size=3,
        wall_size=0.25,
        max_episode_steps=100,
        forward_step_size=0.2,  # range in (0, 1) float
        turn_step_size=90,  # range in (0, 90) int
        obs_name="rgb",
        render_view="agent",
        rnd_init=False,
        rnd_goal=False,
        **kwargs
    ):
        # customize the maze
        self.num_rows = num_rows  # number of rooms on rows (x axis) Note: origin is on the top left.
        self.num_cols = num_cols  # number of rooms on cols (y axis)
        self.room_size = room_size  # size of the room
        self.gap_size = wall_size  # room gap size (i.e. wall thickness)
        # if text map is provided, reset the rows and cols
        try:
            self.array_map, self.valid_locations = self._load_txt()
            assert self.array_map.shape[0] > 2 and self.array_map.shape[1] > 2, \
                print(f"The maze size should be bigger than 2 x 2")
            self.num_rows = self.array_map.shape[0] - 2
            self.num_cols = self.array_map.shape[1] - 2
        except FileNotFoundError:
            print("Text map file is not accessible.")

        # customize the action
        self.ACTION_NAME = ['turn_left', 'turn_right', 'forward', 'backward']  # name of the discrete actions
        self.action_space = spaces.Discrete(len(self.ACTION_NAME))  # create the action space
        self.forward_step_size = forward_step_size  # minimal forward/backward step size
        self.turn_step_size = turn_step_size  # minimal turn left/right step size

        # customizable observations
        self.observation_name = obs_name  # name of the observations
        self.render_view = render_view  # render mode

        # customize start and goal locations
        self.rnd_init = rnd_init  # whether randomize the start location
        self.rnd_goal = rnd_goal  # whether randomize the goal location
        self.start_info = {}  # start information
        self.goal_info = {}  # goal information

        # Debug parameters
        self.step_count = 0
        self.agent = Agent()
        self.entities = []
        self.rooms = []
        self.wall_segs = []
        self.max_forward_step = 0
        self.min_x, self.max_x = 0, 0
        self.min_z, self.max_z = 0, 0

        # construct the domain
        super().__init__(
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

    def reset(self):
        """
            Because I have to add the customizable observations, I override the based method
        """
        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        self.wall_segs = []

        # Generate the world (overridden)
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.rand if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, [
            'sky_color',
            'light_pos',
            'light_color',
            'light_ambient'
        ])

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max('forward_step')

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min([r.min_x for r in self.rooms])
        self.max_x = max([r.max_x for r in self.rooms])
        self.min_z = min([r.min_z for r in self.rooms])
        self.max_z = max([r.max_z for r in self.rooms])

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # render the observation
        obs = self._render_customize_obs()

        return obs

    def step(self, action):
        # save information
        info = {}

        # step counter
        self.step_count += 1

        # set action execution step
        fwd_step = self.forward_step_size
        fwd_drift = 0.0
        turn_step = self.turn_step_size

        # perform the action
        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.move_agent(-fwd_step, fwd_drift)

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)

        # Generate the current camera image
        obs = self._render_customize_obs()

        # compute the reward
        reward = self.compute_reward()

        # If the maximum time step count is reached or the reward is received
        if self.step_count >= self.max_episode_steps or reward == 1.0:
            done = True
        else:
            done = False

        return obs, reward, done, info

    def compute_reward(self):
        # check whether the agent reaches the goal
        dist, _ = self._reach_goal()
        self.goal_info['dist'] = dist

        # return the reward based on the reaching result
        if dist < 0.5:
            return 1
        else:
            return 0

    """ Auxiliary functions
    """
    # check whether the agent reaches the goal
    def _reach_goal(self):
        # check the distance
        agent_pos = self.agent.pos
        agent_ori = self.agent.dir

        goal_pos = self.goal_info['pos']
        goal_ori = self.goal_info['ori']

        # compute the distance
        return np.sum((agent_pos - goal_pos) ** 2), np.abs(agent_ori - goal_ori)

    # agent spawn function
    def _place_agent(self, room=None, pos=None, ori=0):
        return self.place_entity(
                    ent=self.agent,
                    room=room,
                    pos=pos,  # agent position (float)
                    dir=ori   # agent orientation (integer)
                )

    def _place_goal(self, room=None, pos=None, ori=0):
        return self.place_entity(
                    ent=Box(color='yellow'),
                    room=room,
                    pos=pos,
                    dir=ori
                )

    @staticmethod
    # load the map from text file
    def _load_txt():
        # read the map from text file
        with open('./env/maze_test.txt', 'r') as f_in:
            file_data = f_in.readlines()
        f_in.close()

        # preprocess the data: remove '\n' and split strings
        map_data = [l_str.rstrip() for l_str in file_data]

        # get the rows and cols
        rows = len(map_data)
        cols = len(map_data[0])
        print(f"Maze: row num = {rows}, col num = {cols}")
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
                else:
                    # save the corridor cell locations
                    map_valid_locations.append((l_idx - 1, s_idx - 1))
        return map_array, map_valid_locations

    # maze generation function
    def _gen_world(self):
        # customize this function to generate all the rooms
        rows = []

        # initialize all rooms
        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):
                # check if the current cell is a corridor cell
                if (j, i) not in self.valid_locations:
                    row.append(None)
                    continue
                # compute the boundary
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size
                # add the room
                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall'
                )
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

        # place the agent and the goal
        if self.rnd_init and self.rnd_goal:  # both random start and goal
            room_candidates = random.sample(self.valid_locations, 2)
            agent_room_loc = room_candidates[0]
            goal_room_loc = room_candidates[1]
        elif self.rnd_init and not self.rnd_goal:  # random start and fixed goal
            location_copy = cp.copy(self.valid_locations)
            location_copy.pop(-1)  # pop the goal location
            agent_room_loc = random.sample(location_copy, 1)[0]
            goal_room_loc = self.valid_locations[-1]
        elif not self.rnd_init and self.rnd_goal:  # fixed start and random goal
            location_copy = cp.copy(self.valid_locations)
            location_copy.pop(0)  # pop the first location
            agent_room_loc = self.valid_locations[0]
            goal_room_loc = random.sample(location_copy, 1)[0]
        else:  # fixed start and goal
            agent_room_loc = self.valid_locations[0]
            goal_room_loc = self.valid_locations[-1]

        # Debug position
        agent_room_los = (0, 0)
        goal_room_loc = (0, 4)

        # place the agent randomly
        print(f"Agent is spawned in room {agent_room_loc}")
        agent_room = rows[agent_room_loc[0]][agent_room_loc[1]]
        self._place_agent(room=agent_room)
        self.start_info['pos'] = self.agent.pos
        self.start_info['ori'] = self.agent.dir

        # place the goal randomly
        print(f"Goal is spawned in room {goal_room_loc}")
        goal_room = rows[goal_room_loc[0]][goal_room_loc[1]]
        # initialize the goal at the center of the room
        self.goal_info['pos'] = np.array([goal_room.mid_x, 0, goal_room.mid_z])
        self.goal_info['ori'] = 0
        # put the red box in at the goal location
        # self.place_entity(ent=Box(color='red'), pos=self.goal_info['pos'], dir=self.goal_info['ori'])

    # render customizable observations
    def _render_customize_obs(self):
        # render observation
        obs = self.render_obs()

        # render the observation
        if self.observation_name == "depth":  # return depth observation; Shape: H x W
            obs = self.render_depth()
        elif self.observation_name == 'rgb-d':  # return RGB + depth observation; Shape: H x W x 4
            rgb_obs = obs
            d_obs = self.render_depth()
            obs = np.concatenate((rgb_obs, d_obs), axis=2)
        elif self.observation_name == 'rgb':  # return RGB observation; Shape: H x W x 3
            obs = obs
        else:
            raise Exception("Invalid observation name.")

        return obs
