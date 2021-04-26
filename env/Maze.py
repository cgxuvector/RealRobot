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
import random
import copy as cp
from gym import spaces
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.entity import *
from gym_miniworld.opengl import *
import matplotlib.pyplot as plt
from skimage.transform import resize


# class TextMaze(MiniWorldEnv):
#     """
#     3-D Maze environment
#         This 3-D environment is constructed based on maze.py in miniworld.
#         We make the following modifications":
#             - Initialize the maze from text files
#             - Egocentric discrete action space
#                 - Forward, backward, turn left, and turn right.
#                 - Customizable action steps.
#             - Customizable observations
#                 - Front view: RGB images
#                 - Front view: Depth images
#                 - Panorama view: RGB images (4 directions: front/back, left/right)
#     """
#     def __init__(
#         self,
#         text_file=None,
#         room_size=3,
#         wall_size=0.25,
#         max_episode_steps=100,
#         forward_step_size=0.2,  # range in (0, 1) float
#         turn_step_size=90,  # range in (0, 90) int
#         obs_name="rgb",
#         rnd_init=False,
#         rnd_goal=False,
#         **kwargs
#     ):
#         """
#         Initial function. Here are several notes:
#             1. The origin is on the top left same as the image coordinate system.
#             2. Each room can be inferred by (row_idx, col_idx)
#             3. default means inheriting from miniworld.py, added means new adding.
#         :param room_size (default): 3
#         :param wall_size (default): thickness of the wall. For Point2DMaze-like mazes, wall_size = 0.25 as default;
#                                     For DeepMind-like mazes, wall_size = 0.01 as default.
#         :param max_episode_steps (default): Maximal number of steps per episode
#         :param forward_step_size (added): step size for forward/backward actions, range in (0, 1)
#         :param turn_step_size (added): turn angle for turn left/turn right actions, degree number (0, 360)
#         :param obs_name (added): observation name. Supporting rgb, depth, rgb-d, panorama-rgb, panorama-depth
#         :param rnd_init (added): flag for setting the initial agent location randomly
#         :param rnd_goal (added): flag for setting the goal location randomly
#         :param kwargs: Other parameters (default)
#         """
#         # load the text files
#         assert text_file is not None, "No text file is provided."
#         # load the map and valid locations (i.e., empty corridor cells)
#         self.array_map, self.valid_locations = self._load_txt(text_file)
#         # note, in miniworld, there is no out-layer walls
#         assert self.array_map.shape[0] > 2 and self.array_map.shape[1] > 2, \
#             print(f"The maze size should be bigger than 2 x 2")
#
#         # customize the maze
#         self.num_rows = self.array_map.shape[0] - 2  # number of rooms on rows (x axis) Note: origin is on the top left.
#         self.num_cols = self.array_map.shape[1] - 2  # number of rooms on cols (y axis)
#         self.room_size = room_size  # size of the room
#         self.gap_size = wall_size  # room gap size (i.e. wall thickness)
#
#         # customize the action space
#         self.ACTION_NAME = ['turn_left', 'turn_right', 'forward', 'backward']  # name of the discrete actions
#         self.action_space = spaces.Discrete(len(self.ACTION_NAME))  # create the action space
#         self.forward_step_size = forward_step_size  # minimal forward/backward step size
#         self.turn_step_size = turn_step_size  # minimal turn left/right step size
#
#         # customizable observation space
#         self.observation_name = obs_name  # name of the observations
#
#         # customizable render.
#         # Note: we use matplotlib to show the rendered observations
#         self.render_init_marker = True  # mark the initial render
#         self.render_fig = None
#         self.render_arrays = None
#         self.render_artists = None
#
#         # customize start and goal locations
#         self.rnd_init = rnd_init  # whether randomize the start location
#         self.rnd_goal = rnd_goal  # whether randomize the goal location
#         self.start_info = {}  # start information
#         self.goal_info = {}  # goal information
#         self.reach_goal_eps = 1e-3  # epsilon as the goal reaching threshold
#
#         # Parameters for bug-free inherit
#         self.step_count = 0
#         self.agent = Agent()
#         self.entities = []
#         self.rooms = []
#         self.wall_segs = []
#         self.max_forward_step = 0
#         self.min_x, self.max_x = 0, 0
#         self.min_z, self.max_z = 0, 0
#
#         # construct the domain
#         super().__init__(
#             max_episode_steps=max_episode_steps or self.num_rows * self.num_cols * 24,
#             **kwargs
#         )
#
#     def reset(self):
#         """
#             Reset function. Because I customize the observation, there for I override the whole function.
#         """
#         # Step count since episode start
#         self.step_count = 0
#
#         # Create the agent
#         self.agent = Agent()
#
#         # List of entities contained
#         self.entities = []
#
#         # List of rooms in the world
#         self.rooms = []
#
#         # Wall segments for collision detection
#         self.wall_segs = []
#
#         # Generate the world (overridden blew in auxiliary functions)
#         self._gen_world()
#
#         # Check if domain randomization is enabled or not
#         rand = self.rand if self.domain_rand else None
#
#         # Randomize elements of the world (domain randomization)
#         self.params.sample_many(rand, self, [
#             'sky_color',
#             'light_pos',
#             'light_color',
#             'light_ambient'
#         ])
#
#         # Get the max forward step distance
#         self.max_forward_step = self.params.get_max('forward_step')
#
#         # Randomize parameters of the entities
#         for ent in self.entities:
#             ent.randomize(self.params, rand)
#
#         # Compute the min and max x, z extents of the whole floorplan
#         self.min_x = min([r.min_x for r in self.rooms])
#         self.max_x = max([r.max_x for r in self.rooms])
#         self.min_z = min([r.min_z for r in self.rooms])
#         self.max_z = max([r.max_z for r in self.rooms])
#
#         # Generate static data
#         if len(self.wall_segs) == 0:
#             self._gen_static_data()
#
#         # Pre-compile static parts of the environment into a display list
#         self._render_static()
#
#         # render the observation (function is written below in auxiliary functions)
#         obs = self._render_customize_obs()
#
#         return obs
#
#     def step(self, action):
#         """
#             Step function
#         """
#         # save information
#         info = {}
#
#         # step counter
#         self.step_count += 1
#
#         # set action execution step
#         fwd_step = self.forward_step_size
#         fwd_drift = 0.0
#         turn_step = self.turn_step_size
#
#         # perform the action
#         if action == self.actions.move_forward:
#             self.move_agent(fwd_step, fwd_drift)
#
#         elif action == self.actions.move_back:
#             self.move_agent(-fwd_step, fwd_drift)
#
#         elif action == self.actions.turn_left:
#             self.turn_agent(turn_step)
#
#         elif action == self.actions.turn_right:
#             self.turn_agent(-turn_step)
#
#         # Generate the current camera image (rendering using customized observation function)
#         obs = self._render_customize_obs()
#
#         # compute the reward (here we only provide sparse reward 1 for goal and 0 otherwise)
#         reward = self.compute_reward()
#
#         # If the maximum time step count is reached or the reward is received
#         if self.step_count >= self.max_episode_steps or reward == 1.0:
#             done = True
#         else:
#             done = False
#
#         return obs, reward, done, info
#
#     def compute_reward(self):
#         """
#             Reward function:
#             Here we provide one sparse reward function. 1 for reaching the goal and 0 otherwise.
#             The reward is 1 when || agent_pos - goal_pos ||_2 < epsilon. We do not consider
#             direction currently.
#         """
#         # check whether the agent reaches the goal
#         dist, _ = self._reach_goal()
#         self.goal_info['dist'] = dist
#
#         # return the reward based on the reaching result
#         if dist < self.reach_goal_eps:
#             return 1
#         else:
#             return 0
#
#     def render(self, mode='human', close=False, view='agent'):
#         # render the current observation
#         obs = self._render_customize_obs()
#
#         # show the rendered observation
#         if self.observation_name != "panorama-rgb" and self.observation_name != "panorama-depth":
#             if self.render_init_marker:
#                 self.render_fig, self.render_arrays = plt.subplots(1)
#                 self.render_arrays.set_title(self.observation_name)
#                 self.render_arrays.axis("off")
#                 self.render_artists = self.render_arrays.imshow(obs)
#                 self.render_init_marker = False
#             else:
#                 self.render_artists.set_data(obs)
#         else:
#             if self.render_init_marker:
#                 self.render_fig, self.render_arrays = plt.subplots(3, 3)
#                 for i in range(3):
#                     for j in range(3):
#                         self.render_arrays[i, j].axis("off")
#                 self.render_artists = []
#                 top_obs = self.render_top_view()
#                 self.render_arrays[0, 1].set_title("F")
#                 self.render_artists.append(self.render_arrays[0, 1].imshow(obs[0]))
#                 self.render_arrays[1, 0].set_title("L")
#                 self.render_artists.append(self.render_arrays[1, 0].imshow(obs[1]))
#                 self.render_arrays[2, 1].set_title("B")
#                 self.render_artists.append(self.render_arrays[2, 1].imshow(obs[2]))
#                 self.render_arrays[1, 2].set_title("R")
#                 self.render_artists.append(self.render_arrays[1, 2].imshow(obs[3]))
#                 self.render_arrays[1, 1].set_title("Top down")
#                 self.render_artists.append(self.render_arrays[1, 1].imshow(top_obs))
#                 self.render_init_marker = False
#             else:
#                 top_obs = self.render_top_view()
#                 self.render_artists[0].set_data(obs[0])
#                 self.render_artists[1].set_data(obs[1])
#                 self.render_artists[2].set_data(obs[2])
#                 self.render_artists[3].set_data(obs[3])
#                 self.render_artists[4].set_data(top_obs)
#         self.render_fig.canvas.draw()
#         plt.pause(0.01)
#
#     """ Auxiliary functions
#     """
#     def _reach_goal(self):
#         """
#             Compute the relative distance between the agent and the goal.
#             Return distance and direction difference
#         """
#         agent_pos = self.agent.pos
#         agent_ori = self.agent.dir
#
#         goal_pos = self.goal_info['pos']
#         goal_ori = self.goal_info['ori']
#
#         # compute the distance and the orientation difference
#         return np.sum((agent_pos - goal_pos) ** 2), np.abs(agent_ori - goal_ori)
#
#     def _place_agent(self, room=None, pos=None, ori=0):
#         """ Place the agent """
#         return self.place_entity(
#                     ent=self.agent,
#                     room=room,
#                     pos=pos,  # agent position (float)
#                     dir=ori   # agent orientation (integer)
#                 )
#
#     def _place_goal(self, room=None, pos=None, ori=0):
#         """ Place the goal """
#         return self.place_entity(
#                     ent=Box(color='red'),
#                     room=room,
#                     pos=pos,  # goal position
#                     dir=ori   # goal direction
#                 )
#
#     @staticmethod
#     def _load_txt(f_path):
#         """ Load the map from text file. """
#         # read the map from text file
#         with open(f_path, 'r') as f_in:
#             file_data = f_in.readlines()
#         f_in.close()
#
#         # preprocess the data: remove '\n' and split strings
#         map_data = [l_str.rstrip() for l_str in file_data]
#
#         # get the rows and cols
#         rows = len(map_data)
#         cols = len(map_data[0])
#         print(f"Maze size: row num = {rows}, col num = {cols}")
#         # create the array map
#         map_array = np.zeros((rows, cols))
#         # store valid locations
#         map_valid_locations = []
#         # construct the map
#         for l_idx, l in enumerate(map_data):
#             for s_idx, s in enumerate(l):
#                 if s == "*":
#                     # wall cell will be 1 and corridor cell will be 0
#                     map_array[l_idx, s_idx] = 1
#                 else:
#                     # save the corridor cell locations
#                     map_valid_locations.append((l_idx - 1, s_idx - 1))
#         return map_array, map_valid_locations
#
#     def _gen_world(self):
#         """ Generate the maze """
#         # customize this function to generate all the rooms
#         rows = []
#
#         # initialize all rooms
#         # For each row
#         for j in range(self.num_rows):
#             row = []
#
#             # For each column
#             for i in range(self.num_cols):
#                 # check if the current cell is a corridor cell
#                 if (j, i) not in self.valid_locations:
#                     row.append(None)
#                     continue
#                 # compute the boundary
#                 min_x = i * (self.room_size + self.gap_size)
#                 max_x = min_x + self.room_size
#
#                 min_z = j * (self.room_size + self.gap_size)
#                 max_z = min_z + self.room_size
#                 # add the room
#                 room = self.add_rect_room(
#                     min_x=min_x,
#                     max_x=max_x,
#                     min_z=min_z,
#                     max_z=max_z,
#                     wall_tex='brick_wall'
#                 )
#                 row.append(room)
#             rows.append(row)
#
#         # record visited room
#         visited = set()
#
#         # connect the neighbors based on the map info
#         for room_loc in self.valid_locations:
#             # locate the current room
#             room = rows[room_loc[0]][room_loc[1]]
#
#             # add the room
#             visited.add(room)
#
#             # compute the neighbors
#             neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, left, right
#
#             # loop valid neighbors
#             for di, dj in neighbors:
#                 ni = room_loc[1] + di
#                 nj = room_loc[0] + dj
#
#                 # check validation
#                 if nj < 0 or nj >= self.num_rows:
#                     continue
#                 if ni < 0 or ni >= self.num_cols:
#                     continue
#                 if (nj, ni) not in self.valid_locations:
#                     continue
#
#                 neighbor = rows[nj][ni]
#
#                 if neighbor in visited:
#                     continue
#
#                 if di == 0:
#                     self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
#                 elif dj == 0:
#                     self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)
#
#         # place the agent and the goal
#         if self.rnd_init and self.rnd_goal:  # both random start and goal
#             room_candidates = random.sample(self.valid_locations, 2)
#             agent_room_loc = room_candidates[0]
#             goal_room_loc = room_candidates[1]
#         elif self.rnd_init and not self.rnd_goal:  # random start and fixed goal
#             location_copy = cp.copy(self.valid_locations)
#             location_copy.pop(-1)  # pop the goal location
#             agent_room_loc = random.sample(location_copy, 1)[0]
#             goal_room_loc = self.valid_locations[-1]
#         elif not self.rnd_init and self.rnd_goal:  # fixed start and random goal
#             location_copy = cp.copy(self.valid_locations)
#             location_copy.pop(0)  # pop the first location
#             agent_room_loc = self.valid_locations[0]
#             goal_room_loc = random.sample(location_copy, 1)[0]
#         else:  # fixed start and goal
#             agent_room_loc = self.valid_locations[0]
#             goal_room_loc = self.valid_locations[-1]
#
#         # place the agent randomly
#         # print(f"Agent is spawned in room {agent_room_loc}")
#         agent_room = rows[agent_room_loc[0]][agent_room_loc[1]]
#         self._place_agent(room=agent_room, pos=np.array([agent_room.mid_x, 0, agent_room.mid_z]))
#         self.start_info['pos'] = self.agent.pos
#         self.start_info['ori'] = self.agent.dir
#
#         # place the goal randomly
#         # print(f"Goal is spawned in room {goal_room_loc}")
#         goal_room = rows[goal_room_loc[0]][goal_room_loc[1]]
#         # initialize the goal at the center of the room
#         self.goal_info['pos'] = np.array([goal_room.mid_x, 0, goal_room.mid_z])
#         self.goal_info['ori'] = 0
#         # place the goal as a red box
#         self._place_goal(room=goal_room, pos=self.goal_info['pos'])
#
#     # render customizable observations
#     def _render_customize_obs(self):
#         # render observation
#         obs = self.render_obs()
#
#         # render the observation
#         if self.observation_name == "depth":  # return depth observation; Shape: H x W
#             obs = self.render_depth()
#         elif self.observation_name == 'rgb-d':  # return RGB + depth observation; Shape: H x W x 4
#             rgb_obs = obs
#             d_obs = self.render_depth()
#             obs = np.concatenate((rgb_obs, d_obs), axis=2)
#         elif self.observation_name == 'rgb':  # return RGB observation; Shape: H x W x 3
#             obs = obs
#         elif self.observation_name == "panorama-rgb" or self.observation_name == "panorama-depth":
#             # render panoramic observation
#             obs = []
#             current_dir = cp.copy(self.agent.dir)
#             for i in range(4):
#                 select_dir = current_dir + 90 * i * np.pi / 180
#                 tmp_obs = self._render_dir_obs(radian_dir=select_dir)
#                 obs.append(tmp_obs)
#             # reset the direction
#             self.agent.dir = current_dir
#         else:
#             raise Exception("Invalid observation name.")
#
#         return obs
#
#     # render panorama view
#     def _render_dir_obs(self, radian_dir=0, frame_buffer=None):
#         """
#             Render an observation from a specific direction
#         """
#         if frame_buffer == None:
#             frame_buffer = self.obs_fb
#
#         # Switch to the default OpenGL context
#         # This is necessary on Linux Nvidia drivers
#         self.shadow_window.switch_to()
#
#         # Bind the frame buffer before rendering into it
#         frame_buffer.bind()
#
#         # Clear the color and depth buffers
#         glClearColor(*self.sky_color, 1.0)
#         glClearDepth(1.0)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#
#         # Set the projection matrix
#         glMatrixMode(GL_PROJECTION)
#         glLoadIdentity()
#         gluPerspective(
#             self.agent.cam_fov_y,
#             frame_buffer.width / float(frame_buffer.height),
#             0.04,
#             100.0
#         )
#
#         # set the render direction
#         self.agent.dir = radian_dir
#
#         # Setup the camera
#         glMatrixMode(GL_MODELVIEW)
#         glLoadIdentity()
#         gluLookAt(
#             # Eye position
#             *self.agent.cam_pos,
#             # Target
#             *(self.agent.cam_pos + self.agent.cam_dir),
#             # Up vector
#             0, 1.0, 0.0
#         )
#
#         if self.observation_name == "panorama-rgb":
#             return self._render_world(
#                 frame_buffer,
#                 render_agent=False
#             )
#         else:
#             self.render_obs(frame_buffer)
#             return frame_buffer.get_depth_map(0.04, 100.0)


# Todo future: Better implementation
class GoalTextMaze(MiniWorldEnv):
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
                - Panorama view: RGB images (4 directions: front/back, left/right)
    """
    def __init__(
        self,
        text_file=None,
        room_size=3,
        wall_size=0.25,
        max_episode_steps=100,
        forward_step_size=0.5,  # range in (0, 1) float
        turn_step_size=90,  # range in (0, 90) int
        obs_name="rgb",
        rnd_init=False,
        rnd_goal=False,
        goal_reach_eps=1e-3,  # values indicates the goal is reached.
        **kwargs
    ):
        """
        Initial function. Here are several notes:
            1. The origin is on the top left same as the image coordinate system.
            2. Each room can be inferred by (row_idx, col_idx)
            3. default means inheriting from miniworld.py, added means new adding.
        :param room_size (default): 3
        :param wall_size (default): thickness of the wall. For Point2DMaze-like mazes, wall_size = 0.25 as default;
                                    For DeepMind-like mazes, wall_size = 0.01 as default.
        :param max_episode_steps (default): Maximal number of steps per episode
        :param forward_step_size (added): step size for forward/backward actions, range in (0, 1)
        :param turn_step_size (added): turn angle for turn left/turn right actions, degree number (0, 360)
        :param obs_name (added): observation name. Supporting rgb, depth, rgb-d, panorama-rgb, panorama-depth, state
        :param rnd_init (added): flag for setting the initial agent location randomly
        :param rnd_goal (added): flag for setting the goal location randomly
        :param kwargs: Other parameters (default)
        """
        # load the text files
        assert text_file is not None, "No text file is provided."
        # load the map and valid locations (i.e., empty corridor cells)
        self.array_map, self.valid_locations = self._load_txt(text_file)
        # note, in miniworld, there is no out-layer walls
        assert self.array_map.shape[0] > 2 and self.array_map.shape[1] > 2, \
            print(f"The maze size should be bigger than 2 x 2")

        # customize the maze
        self.num_rows = self.array_map.shape[0] - 2  # number of rooms on rows (x axis) Note: origin is on the top left.
        self.num_cols = self.array_map.shape[1] - 2  # number of rooms on cols (y axis)
        self.room_size = room_size  # size of the room
        self.gap_size = wall_size  # room gap size (i.e. wall thickness)

        # customize the action space
        self.ACTION_NAME = ['turn_left', 'turn_right', 'forward', 'backward']  # name of the discrete actions
        self.action_space = spaces.Discrete(len(self.ACTION_NAME))  # create the action space
        self.forward_step_size = forward_step_size  # minimal forward/backward step size
        self.turn_step_size = turn_step_size  # minimal turn left/right step size

        # customizable observation space
        self.observation_name = obs_name  # name of the observations

        # customizable render.
        # Note: we use matplotlib to show the rendered observations
        self.render_init_marker = True  # mark the initial render
        self.render_fig = None
        self.render_arrays = None
        self.render_artists = None

        # customize start and goal locations
        self.rnd_init = rnd_init  # whether randomize the start location
        self.rnd_goal = rnd_goal  # whether randomize the goal location
        self.start_info = {}  # start information
        self.goal_info = {}  # goal information
        self.reach_goal_eps = goal_reach_eps  # epsilon as the goal reaching threshold

        # Parameters for bug-free inherit
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
            max_episode_steps=max_episode_steps or self.num_rows * self.num_cols * 24,
            **kwargs
        )

    def reset(self):
        """
            Reset function. Because I customize the observation, there for I override the whole function.
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

        # Check if domain randomization is enabled or not
        rand = self.rand if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, [
            'sky_color',
            'light_pos',
            'light_color',
            'light_ambient'
        ])

        # Generate the world (overridden blew in auxiliary functions)
        self._gen_world()

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

        # set the agent to the goal room
        goal_room = self.goal_info['room']
        self._place_agent(goal_room, pos=np.array([goal_room.mid_x, 0, goal_room.mid_z]))
        # retrieve the observation of the goal
        self.goal_info['goal_obs'] = self._render_customize_obs()

        # reset the agent to the start room
        start_room = self.start_info['room']
        # retrieve the observation
        self._place_agent(start_room, pos=np.array([start_room.mid_x, 0, start_room.mid_z]))
        obs = self._render_customize_obs()

        # construct the goal-rl observation
        return {'observation': obs, 'achieved_goal': obs, 'desired_goal': self.goal_info['goal_obs']}

    def step(self, action):
        """
            Step function
        """
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

        # Generate the current camera image (rendering using customized observation function)
        obs = self._render_customize_obs()

        # compute the reward (here we only provide sparse reward 1 for goal and 0 otherwise)
        reward = self.compute_reward()

        # If the maximum time step count is reached or the reward is received
        if self.step_count >= self.max_episode_steps or reward == 1.0:
            done = True
        else:
            done = False

        # construct the goal-rl observation
        obs = {'observation': obs, 'achieved_goal': obs, 'desired_goal': self.goal_info['goal_obs']}

        return obs, reward, done, info

    def compute_reward(self):
        """
            Reward function:
            Here we provide one sparse reward function. 1 for reaching the goal and 0 otherwise.
            The reward is 1 when || agent_pos - goal_pos ||_2 < epsilon. We do not consider
            direction currently.
        """
        # check whether the agent reaches the goal
        dist, _ = self._reach_goal()
        self.goal_info['dist'] = dist

        # return the reward based on the reaching result
        if dist < self.reach_goal_eps:
            return 0
        else:
            return -1

    def render(self, mode='human', close=False, view='agent'):
        # render the current observation
        obs = self._render_customize_obs()

        # no need to visualize state
        if self.observation_name == "state":
            return obs

        # show the rendered observation
        if self.observation_name != "panorama-rgb" and self.observation_name != "panorama-depth":
            if self.render_init_marker:
                self.render_fig, self.render_arrays = plt.subplots(1)
                self.render_arrays.set_title(self.observation_name)
                self.render_arrays.axis("off")
                self.render_artists = self.render_arrays.imshow(obs)
                self.render_init_marker = False
            else:
                self.render_artists.set_data(obs)
        else:
            if self.render_init_marker:
                self.render_fig, self.render_arrays = plt.subplots(3, 3)
                for i in range(3):
                    for j in range(3):
                        self.render_arrays[i, j].axis("off")
                self.render_artists = []
                top_obs = self.render_top_view()
                self.render_arrays[0, 1].set_title("F")
                self.render_artists.append(self.render_arrays[0, 1].imshow(obs[0]))
                self.render_arrays[1, 0].set_title("L")
                self.render_artists.append(self.render_arrays[1, 0].imshow(obs[1]))
                self.render_arrays[2, 1].set_title("B")
                self.render_artists.append(self.render_arrays[2, 1].imshow(obs[2]))
                self.render_arrays[1, 2].set_title("R")
                self.render_artists.append(self.render_arrays[1, 2].imshow(obs[3]))
                self.render_arrays[1, 1].set_title("Top down")
                self.render_artists.append(self.render_arrays[1, 1].imshow(top_obs))
                self.render_init_marker = False
            else:
                top_obs = self.render_top_view()
                self.render_artists[0].set_data(obs[0])
                self.render_artists[1].set_data(obs[1])
                self.render_artists[2].set_data(obs[2])
                self.render_artists[3].set_data(obs[3])
                self.render_artists[4].set_data(top_obs)
        self.render_fig.canvas.draw()
        plt.pause(0.01)

    """ Auxiliary functions
    """
    def _reach_goal(self):
        """
            Compute the relative distance between the agent and the goal.
            Return distance and direction difference
        """
        agent_pos = self.agent.pos
        agent_ori = self.agent.dir

        goal_pos = self.goal_info['pos']
        goal_ori = self.goal_info['ori']

        # compute the distance and the orientation difference
        return np.sum((agent_pos - goal_pos) ** 2), np.abs(agent_ori - goal_ori)

    def _place_agent(self, room=None, pos=None, ori=0):
        """ Place the agent """
        return self.place_entity(
                    ent=self.agent,
                    room=room,
                    pos=pos,  # agent position (float)
                    dir=ori   # agent orientation (integer)
                )

    def _place_goal(self, room=None, pos=None, ori=0):
        """ Place the goal """
        return self.place_entity(
                    ent=Box(color='red'),
                    room=room,
                    pos=pos,  # goal position
                    dir=ori   # goal direction
                )

    @staticmethod
    def _load_txt(f_path):
        """ Load the map from text file. """
        # read the map from text file
        with open(f_path, 'r') as f_in:
            file_data = f_in.readlines()
        f_in.close()

        # preprocess the data: remove '\n' and split strings
        map_data = [l_str.rstrip() for l_str in file_data]

        # get the rows and cols
        rows = len(map_data)
        cols = len(map_data[0])
        # print(f"Maze size: row num = {rows}, col num = {cols}")
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

    def _gen_world(self):
        """ Generate the maze """
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

        # place the agent randomly
        # print(f"Agent is spawned in room {agent_room_loc}")
        start_room = rows[agent_room_loc[0]][agent_room_loc[1]]
        self._place_agent(room=start_room, pos=np.array([start_room.mid_x, 0, start_room.mid_z]))
        self.start_info['pos'] = self.agent.pos
        self.start_info['ori'] = self.agent.dir
        self.start_info['room'] = start_room

        # place the goal randomly
        # print(f"Goal is spawned in room {goal_room_loc}")
        goal_room = rows[goal_room_loc[0]][goal_room_loc[1]]
        # initialize the goal at the center of the room
        self.goal_info['pos'] = np.array([goal_room.mid_x, 0, goal_room.mid_z])
        self.goal_info['ori'] = 0
        self.goal_info['room'] = goal_room
        # place the goal as a red box
        # self._place_goal(room=goal_room, pos=self.goal_info['pos'])

    # render customizable observations
    def _render_customize_obs(self):
        # render observation
        obs = self.render_obs()

        # render the observation
        if self.observation_name == "depth":  # return depth observation; Shape: H x W
            obs = self.render_depth()
            obs = resize(obs, (32, 32))
        elif self.observation_name == 'rgb-d':  # return RGB + depth observation; Shape: H x W x 4
            rgb_obs = obs
            rgb_obs = resize(rgb_obs, (32, 32))
            d_obs = self.render_depth()
            d_obs = resize(d_obs, (32, 32))
            obs = np.concatenate((rgb_obs, d_obs), axis=2)
        elif self.observation_name == 'rgb':  # return RGB observation; Shape: H x W x 3
            obs = obs
            obs = resize(obs, (32, 32))
        elif self.observation_name == "panorama-rgb" or self.observation_name == "panorama-depth":
            # render panoramic observation
            obs = []
            current_dir = cp.copy(self.agent.dir)
            for i in range(4):
                select_dir = current_dir + 90 * i * np.pi / 180
                tmp_obs = self._render_dir_obs(radian_dir=select_dir)
                tmp_obs = resize(tmp_obs, (32, 32))
                obs.append(tmp_obs)
            # reset the direction
            self.agent.dir = current_dir
        elif self.observation_name == "state":
            obs = np.array([self.agent.pos[0], self.agent.pos[2], self.agent.dir])
        else:
            raise Exception("Invalid observation name.")

        return obs

    # render panorama view
    def _render_dir_obs(self, radian_dir=0, frame_buffer=None):
        """
            Render an observation from a specific direction
        """
        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0
        )

        # set the render direction
        self.agent.dir = radian_dir

        # Setup the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.agent.cam_pos,
            # Target
            *(self.agent.cam_pos + self.agent.cam_dir),
            # Up vector
            0, 1.0, 0.0
        )

        if self.observation_name == "panorama-rgb":
            return self._render_world(
                frame_buffer,
                render_agent=False
            )
        else:
            self.render_obs(frame_buffer)
            return frame_buffer.get_depth_map(0.04, 100.0)

    def plot_goal_obs(self):
        obs = self.goal_info['goal_obs']
        top_down_obs = self.render_top_view()
        fig, arrays = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                arrays[i, j].axis("off")
        arrays[0, 1].set_title("F")
        arrays[0, 1].imshow(obs[0])
        arrays[1, 0].set_title("L")
        arrays[1, 0].imshow(obs[1])
        arrays[2, 1].set_title("B")
        arrays[2, 1].imshow(obs[2])
        arrays[1, 2].set_title("R")
        arrays[1, 2].imshow(obs[3])
        arrays[1, 1].set_title('top down')
        arrays[1, 1].imshow(top_down_obs)
        plt.show()

    def plot_panorama_obs(self, obs):
        top_down_obs = self.render_top_view()
        fig, arrays = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                arrays[i, j].axis("off")
        arrays[0, 1].set_title("F")
        arrays[0, 1].imshow(obs[0])
        arrays[1, 0].set_title("L")
        arrays[1, 0].imshow(obs[1])
        arrays[2, 1].set_title("B")
        arrays[2, 1].imshow(obs[2])
        arrays[1, 2].set_title("R")
        arrays[1, 2].imshow(obs[3])
        arrays[1, 1].set_title('top down')
        arrays[1, 1].imshow(top_down_obs)
        plt.show()
