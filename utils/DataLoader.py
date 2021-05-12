import json
import os
import random
import numpy as np
import copy as cp
import multiprocessing as mp
from skimage.transform import resize
import IPython.terminal.debugger as Debug

""" Architecture of the returned training and testing dataset
        - This class creates an object that splits the mazes into 4 splits for 4-fold cross validation
        - The split datasets are stored in two dicts named self.trn_dataset amd self.tst_dataset, respectively.
        - For each dataset: e.g., self.trn_dataset
            - self.trn_dataset['split_1'] is a list of dicts. Each element is a dict that contains the info of one maze.
            e.g., self.trn_dataset['split_1'][0] is a maze dict that contains {id: maze id, map: maze map, data: data}
            Note: id is an integer, map is a list, data is also a list (transition or episode)
"""


class DataLoader(object):
    def __init__(self, data_root_dir, data_file_name, target_data_type, maze_idx_list):
        # parameters
        self._root_dir = data_root_dir
        self._file_name = data_file_name

        # dataset sample type
        self._data_type = target_data_type

        # maze used to construct the dataset
        self._use_maze_idx = maze_idx_list

        # data storage
        self._raw_data = []  # store the raw data
        self._dataset = []  # dataset after preporocess the raw data
        # dataset architecture, each is a dict
        # - 'split_1', 'split_2', 'split_3', 'split_4'
        # each 'split_1' is a list that contains the data for each maze
        # each maze data is a dict that contains 'idx', 'data', 'map'
        # one data can be accessed by self.trn_dataset['split_1'][0]['data']
        self.trn_dataset = {'split_1': [], 'split_2': [], 'split_3': [], 'split_4': []}  # training dataset
        self.tst_dataset = {'split_1': [], 'split_2': [], 'split_3': [], 'split_4': []}  # testing dataset

        # init the data loader
        self.load_data()
        self.gen_dateset()
        self.split_data()

    def load_data(self):
        """ Load the initial data """
        target_file = os.path.join(self._root_dir, self._file_name)
        with open(target_file, "r") as f_in:
            data = json.load(f_in)
        f_in.close()

        # select the target mazes
        for item in data:
            if item['id'] in self._use_maze_idx:
                self._raw_data.append(item)

    def split_data(self):
        # split the data into four splits
        random.seed(1234)
        # split the mazes into four splits
        maze_indices = cp.copy(self._use_maze_idx)
        maze_num_per_split = len(self._use_maze_idx) // 4
        idx_split_1 = random.sample(maze_indices, maze_num_per_split)
        [maze_indices.remove(d) for d in idx_split_1]
        idx_split_2 = random.sample(maze_indices, maze_num_per_split)
        [maze_indices.remove(d) for d in idx_split_2]
        idx_split_3 = random.sample(maze_indices, maze_num_per_split)
        [maze_indices.remove(d) for d in idx_split_3]
        idx_split_4 = maze_indices

        # split the data
        all_splits_data = {'split_1': [], 'split_2': [], 'split_3': [], 'split_4': []}
        data = self._dataset
        for d in data:
            if d['id'] in idx_split_1:
                all_splits_data['split_1'].append(d)
            elif d['id'] in idx_split_2:
                all_splits_data['split_2'].append(d)
            elif d['id'] in idx_split_3:
                all_splits_data['split_3'].append(d)
            else:
                all_splits_data['split_4'].append(d)

        # store the training and testing data
        for tst_key in all_splits_data.keys():
            # save the hyper_model_with_auto_mask dataset
            self.tst_dataset[tst_key] = all_splits_data[tst_key]
            # save the training dataset
            for trn_key in all_splits_data.keys():
                if trn_key == tst_key:
                    continue
                else:
                    self.trn_dataset[tst_key] += all_splits_data[trn_key]

    def gen_dateset(self):
        # create the queue
        queue = mp.Queue()

        # create the processes
        if self._data_type == "transition":
            processes = [mp.Process(target=self._episode2transition, args=(m_d, queue)) for m_d in self._raw_data]
        elif self._data_type == "episode":
            processes = [mp.Process(target=self._episode2trajectory, args=(m_d, queue)) for m_d in self._raw_data]
        else:
            raise Exception("Invalid data type. Expected transition or episode")

        # start the processes
        [p.start() for p in processes]

        # retrieve all the data
        dataset = [queue.get() for p in processes]

        # wait for all the processes terminate
        [p.join() for p in processes]

        # close all processes
        [p.close() for p in processes]

        # reconstruct the dataset
        for d in dataset:
            self._dataset.append({'id': d[0], 'map': d[1], 'data': d[2:]})

    @staticmethod
    def _gen_goal_map_image(g, m, img_size):
        """
        Function is used to convert goal position and map to a fixed size images
        :param g: goal location in the maze
        :param m: 2-D rough map
        :param img_size: size of the image
        :return: goal and map image
        Note: the goal is indicated by values > 0
        """
        # create the goal array
        goal_arr = np.zeros_like(np.array(m))
        goal_arr[g[0] // 5, g[1] // 5] = 1
        # convert to images
        goal_img = resize(goal_arr, img_size)
        map_img = resize(1 - np.array(m), img_size)
        # # show subplots
        # fig, ax = plt.subplots(1, 2)
        # ax[0].set_title(f"({g[0]//5}, {g[1]//5})")
        # ax[0].imshow(goal_img)
        # ax[1].set_title(f"{g}")
        # ax[1].imshow(map_img)
        # plt.show()
        return goal_img, map_img

    @staticmethod
    def _episode2transition(maze_data, mp_q):
        """ Convert a single episode into transitions
            The logic is as follows:
                episode = [s_0, a_1, s_1, a_2, s_2, ...., s_T-1, a_t, s_T(g)]
                trans = [state, action, next state, goal, map]
        """
        # tmp dataset
        trans_data_tmp = []
        # we got the basic components
        m_episodes = maze_data['episodes']
        m_map = maze_data['map']
        m_idx = maze_data['id']
        # load the episodes
        trans_data_tmp.append(m_idx)  # save the maze idx as the first component
        trans_data_tmp.append(m_map)  # save the maze map as the second component
        for episode in m_episodes:
            # goal locations
            goal_loc = episode[-1]
            # construct transitions
            state = episode[0]  # starts from s_0
            for i in range(2, len(episode), 2):  # i starts from i = 2 <=> s_1
                action = episode[i - 1]
                next_state = episode[i]
                # construct a transition
                trans = [state, action, next_state, goal_loc, m_map]
                trans_data_tmp.append(trans)
                state = next_state

        mp_q.put(trans_data_tmp)  # push the data to queue for multiple processes

    @staticmethod
    def _episode2trajectory(maze_data, mp_q):
        # tmp dataset
        trajs_data_tmp = []
        # we got the basic components
        m_episodes = maze_data['episodes']
        m_map = maze_data['map']
        m_idx = maze_data['id']
        # load the episodes
        trajs_data_tmp.append(m_idx)  # save the maze idx as the first component
        trajs_data_tmp.append(m_map)  # save the maze map as the second component

        trajs_data_tmp.append(m_episodes)  # save the maze episodes as the third component

        mp_q.put(trajs_data_tmp)  # push the data to queue for multiple processes


# hyper_model_with_auto_mask code
# my_loader = DataLoader('../data', 'maze_11x11_0-9.json', 'transition', [0, 1, 2, 3])
# print("Print info")
# print("Training mazes id: ")
# for key in my_loader.trn_dataset.keys():
#    for item in my_loader.trn_dataset[key]:
#        print(item['id'])
#    print("-----")

#print("Testing mazes id: ")
#for key in my_loader.tst_dataset.keys():
#    print(my_loader.tst_dataset[key][0]['id'])
