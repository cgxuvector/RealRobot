import torch
import os
import argparse
import numpy as np
from multi_goal_model.DeepNetwork import HyperNetwork, DynamicModel
from multi_goal_model.DeepNetwork import DynaFilterNet
from multi_goal_model.Attention import UNet
from utils.TreeSearchAlg.AStar import Node
from utils import MapProcessor
import matplotlib.pyplot as plt
import random

import IPython.terminal.debugger as Debug


# inherit from Node, adding one attribute action
class ActNode(Node):
    def __init__(self, parent=None, position=None, act=None):
        super(ActNode, self).__init__(parent, position)
        self.act = act


# dynamic model with hypernetwork map representation
class HyperModel(object):
    def __init__(self, configs):
        # set the device
        self.device = torch.device(configs["device"])

        # set the configurations
        self.configs = configs

        # load the model
        self.hyper_model, self.dyna_model = self.load_model()

        self.walls_locs = []

        # map_filter model
        if self.configs['map_filter_path']:
            if self.configs['map_mask'] == "auto_mask":
                self.map_filter = DynaFilterNet(self.configs['maze_size']).to(self.device)
            elif self.configs['map_mask'] == "auto_mask_unet":
                self.map_filter = UNet().to(self.device)

            self.map_filter.load_state_dict(
                torch.load(os.path.join(self.configs['map_filter_path'], 'best_trn_dyna_filter_model.pt'),
                           map_location=self.configs['device']))
            self.map_filter.eval()

    def load_model(self):
        # define the models
        hyper_model = HyperNetwork(use_ego_motion=True).to(self.device)
        dyna_model = DynamicModel().to(self.device)

        # load the trained parameters
        hyper_model.load_state_dict(torch.load(os.path.join(self.configs['model_path'], 'trn_hyper_model.pt'),
                                               map_location="cuda:0"))
        hyper_model.eval()

        return hyper_model, dyna_model

    # Modified A star planner for learned models
    def modified_a_star_planner(self, state, goal, dyna_model, weights=None):
        # convert to numpy
        start = state.numpy()  # current state
        end = goal.numpy().squeeze(0)  # goal state

        # action list
        actions_list = [0, 1, 2, 3]  # the action space should be input before the planning happens

        # Create start and end node (initialize)
        start_node = ActNode(None, start.astype(np.int).tolist()[0])
        start_node.g = start_node.h = start_node.f = 0
        end_node = ActNode(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []  # contains the node that need to be explored
        closed_list = []  # contains the node that are already explored

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # obtain the node with the least f value in open list
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # check if we reach the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    # append the location
                    path.append(current.position)
                    # append the action
                    if current.act is not None:
                        path.append(current.act)
                    current = current.parent
                return path[::-1]  # Return reversed path

            # remove the current node from the open list
            open_list.pop(current_index)
            # add the current node into closed list
            closed_list.append(current_node)

            # Generate neighbors
            for act in actions_list:  # Adjacent squares
                # convert states to tensors
                state_tensor = torch.tensor(current_node.position).view(1, 2).float().to(self.device)
                action_tensor = torch.tensor([act]).float().to(self.device)

                # Predict the next state using the learned model
                with torch.no_grad():
                    # predict the offset
                    node_pos_offset = dyna_model(state_tensor, action_tensor, weights)
                    # compute the next state
                    node_pos = state_tensor + node_pos_offset
                    # round the results for A star planner
                    tmp_node_pos = np.round(node_pos.cpu().numpy().squeeze(0)).astype(np.int).tolist()

                # only for error tolerated planning
                if self.configs["use_model_error_tolerated_planner"]:
                    if tmp_node_pos in self.walls_locs:
                        continue

                # create a candidate node
                new_node = ActNode(current_node, tmp_node_pos, act=act)

                # if current node is already searched
                in_close_flag = False
                for closed_child in closed_list:
                    if new_node == closed_child:
                        in_close_flag = True
                        break
                if in_close_flag:
                    continue

                # compute the measures of current node
                new_node.g = current_node.g + 1
                # Manhattan distance
                new_node.h = abs(new_node.position[0] - end[0]) + abs(new_node.position[1] - end[1])
                # Euclidean distance
                # new_node.h = np.sqrt((new_node.position[0] - end[0]) ** 2 + (new_node.position[1] - end[1]) ** 2)
                new_node.f = new_node.g + new_node.h

                # check whether add it into the open list
                in_open_flag = False
                for open_node in open_list:
                    if not (open_node == new_node):
                        continue
                    else:
                        # if in the open list and have smaller cost
                        in_open_flag = True
                        if open_node.f > new_node.f:
                            open_list.append(new_node)

                # if not in the open list then add it
                if not in_open_flag:
                    open_list.append(new_node)

    def get_action(self, inputs):
        # unroll the input
        s, g, m, walls = inputs

        if self.configs["use_model_error_tolerated_planner"]:
            self.walls_locs = walls.copy()

        # preprocess the state to tensor
        s = torch.tensor(s).view(1, -1).float()
        g = torch.tensor(g).view(1, -1).float()

        # preprocess the map
        if self.configs['map_feature'] == "flatten":
            m = torch.tensor(m).unsqueeze(dim=0).float()
        else:
            # for conv2d, it is clear that we set the empty place to be 1 to highlight the feature
            m = np.ones_like(m) - m
            m = MapProcessor.resize(m, (32, 32))
            m = torch.tensor(m).unsqueeze(dim=0).unsqueeze(dim=0).float().to(self.device)

        # generate the weights
        weights = self.hyper_model(m)

        # plan the path
        path = self.modified_a_star_planner(s, g, self.dyna_model, weights)

        # no path found
        if path is None:
            print(f"Fail to find the path: {path}")
            pred_a = None
            pred_next_state = None
        else:
            pred_a = path[1]  # get the action planned from the path
            pred_next_state = path[2]  # get the predicted next state from the path

        return pred_a, pred_next_state

    # rewrite the function for ego_motion
    def plot_prediction_error_heat_map_ego_motion(self, m_id=None):
        # define the step function as an inner function
        def inner_step_func(s, a, m, o):
            # copy
            next_s = s.copy()
            # action list
            if a == 0:  # turn left
                o += np.pi / 2
                next_s = next_s[0:2] + [np.sin(o), np.cos(o)]
            elif a == 1:  # turn right
                o -= np.pi / 2
                next_s = next_s[0:2] + [np.sin(o), np.cos(o)]
            elif a == 2:  # forward
                # determine the orientation
                if o == 0:  # east
                    tmp_pos = [next_s[0] + 0.5, next_s[1], np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                elif o == np.pi / 2:  # north
                    tmp_pos = [next_s[0], next_s[1] - 0.5, np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                elif o == np.pi:  # west
                    tmp_pos = [next_s[0] - 0.5, next_s[1], np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                elif o == 3 * np.pi / 2:  # south
                    tmp_pos = [next_s[0], next_s[1] + 0.5, np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                else:
                    raise Exception(f"Wrong orientation. {o}")
            elif a == 3:  # backward
                # determine the orientation
                if o == 0:  # east
                    tmp_pos = [next_s[0] - 0.5, next_s[1], np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                elif o == np.pi / 2:  # north
                    tmp_pos = [next_s[0], next_s[1] + 0.5, np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                elif o == np.pi:  # west
                    tmp_pos = [next_s[0] + 0.5, next_s[1], np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                elif o == 3 * np.pi / 2:  # south
                    tmp_pos = [next_s[0], next_s[1] - 0.5, np.sin(o), np.cos(o)]
                    next_s = tmp_pos if m[int(tmp_pos[1] // 0.5)][int(tmp_pos[0] // 0.5)] == 1 else next_s
                else:
                    raise Exception(f"Wrong orientation. {o}")
            else:
                raise Exception("Error")

            return next_s

        # set the maze id
        self.configs["maze_id"] = m_id

        # load the rough map
        with open(f'./env/mazes/maze_{self.configs["maze_size"]}_{self.configs["maze_id"]}.txt') as f_in:
            map_lines = f_in.readlines()
        f_in.close()

        # convert to map numpy
        map_data = np.array([[int(float(item)) for item in l.rstrip().split(",")] for l in map_lines])
        map_data = np.where(map_data == 0.0, 1.0, map_data)
        map_data = np.where(map_data == 2.0, 0.0, map_data)

        # find the valid positions on the map
        rows, cols = np.where(map_data == 1.0)
        map_state_list = [[r, c] for r, c in zip(rows, cols)]

        # create an empty heatmap
        maze_arr = np.zeros(((self.configs['maze_size'] - 3) * 6 + 7, (self.configs['maze_size'] - 3) * 6 + 7))

        # action names
        action_names = ['turn_left', 'turn_right', 'forward', 'backward']

        for state in map_state_list:
            # compute the current pos in the maze
            r, c = state
            # loop all the possible pos corresponding to the map state
            # Note: 7 is the number of possible position in one room in one direction
            #       3 is the size of the room.
            for i in range(7):
                # compute the y position
                y = (r - 1) * 3 + i * 0.5
                for j in range(7):
                    # compute the x position
                    x = (c - 1) * 3 + j * 0.5

                    # check the validation of the position
                    if x % 3 == 0 and y % 3 == 0:
                        row_idx = int(y // 3)
                        col_idx = int(x // 3)
                        if map_data[row_idx][col_idx] == 0 or map_data[row_idx][col_idx+1] == 0 or map_data[row_idx+1][col_idx] == 0 or map_data[row_idx+1][col_idx+1] == 0:
                            # print(f"Map state = {state}, Invalid pos = ({x}, {y}),
                            # heatmap pos = ({y // 0.5}, {x // 0.5})")
                            continue
                    if x % 3 == 0:
                        col_idx = int(x // 3)
                        if map_data[r][col_idx] == 0 or map_data[r][col_idx + 1] == 0:
                            # print(f"Map state = {state}, Invalid pos = ({x}, {y}),
                            # heatmap pos = ({y // 0.5}, {x // 0.5})")
                            continue
                    if y % 3 == 0:
                        row_idx = int(y // 3)
                        if map_data[row_idx][c] == 0 or map_data[row_idx + 1][c] == 0:
                            # print(f"Map state = {state}, Invalid pos = ({x}, {y}),
                            # heatmap pos = ({y // 0.5}, {x // 0.5})")
                            continue

                    # set the maze array
                    maze_arr[int(y // 0.5)][int(x // 0.5)] = 1.0

        heatmap_list = [np.zeros(((self.configs['maze_size'] - 3) * 6 + 7, (self.configs['maze_size'] - 3) * 6 + 7)) for
                        _ in range(4)]

        rows, cols = np.where(maze_arr == 1.0)
        maze_state_list = [[r, c] for r, c in zip(rows, cols)]
        for state in maze_state_list:
            # compute the current pos in the maze
            r, c = state
            # compute the location
            valid_position = [c * 0.5, r * 0.5]

            # four orientations
            act = self.configs['action']
            for i, ori in enumerate([0, np.pi / 2, np.pi, 3 * np.pi / 2]):
                # compute the location
                encode_ori = [np.sin(ori), np.cos(ori)]
                loc = valid_position + encode_ori

                # predict the next state
                state_tensor = torch.tensor(loc).view(1, -1).float().to(self.device)
                act_tensor = torch.tensor(act).view(1, -1).float().to(self.device)

                # manually mask the map
                map_copy = None
                m, _ = MapProcessor.manual_mask_ego_motion(loc, map_data)
                map_copy = m.copy()
                m = MapProcessor.resize_optim(m, (32, 32))
                map_tensor = torch.tensor(m).unsqueeze(dim=0).unsqueeze(dim=0).float().to(self.device)
                # generate the mazes
                weights = self.hyper_model(map_tensor)
                pred_next_offset = self.dyna_model(state_tensor, act_tensor, weights)
                pred_next = (state_tensor + pred_next_offset).detach().cpu().numpy()

                # compute the ground truth next state
                next_state = inner_step_func(loc, act, maze_arr, ori)
                positional_err = np.sqrt(np.sum(np.array(next_state)[0:2] - pred_next[0, 0:2])**2)
                rotational_err = np.sqrt(np.sum(np.array(next_state)[2:] - pred_next[0, 2:])**2)
                if positional_err > 0.25 or rotational_err > 0.25:
                    print(f"state = {loc}, act = {action_names[act]}, next state = {next_state}, pred_next = {pred_next},"
                          f"P err = {positional_err}, R err = {rotational_err}")

                heatmap_list[i][r, c] = positional_err + rotational_err

        return map_data, maze_arr, heatmap_list


def parse_input():
    parser = argparse.ArgumentParser()
    # model to evaluate
    parser.add_argument("--model_type", type=str, default="hyper_model")
    # data to evaluate
    parser.add_argument("--eval_data", type=str, default="env",
                        help="valid input: transition, episode, or env")

    # environment to evaluate
    # maze size
    parser.add_argument("--maze_size", type=int, default=7)
    # maze id
    parser.add_argument("--maze_id", type=int, default=0)
    # maximal episode steps
    parser.add_argument("--max_episode_steps", type=int, default=50)

    # evaluate split
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="./results/from_panzer/hyper_test/"
                                                          "multi_mazes_7_split_0_act3/05-14/"
                                                          "11-00-33_multi_mazes_7_split_0_act3_batch_64/model")

    # evaluation configurations
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_type", type=str, default="tst")
    parser.add_argument("--map_feature", type=str, default="conv2d")
    parser.add_argument("--eval_episode_num", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda:0")

    # for heat map
    parser.add_argument("--action", type=int, default=2)

    # for planning having tolerance of model error
    # There is not only one way to do this. Here, we just use a list to track the wall positions
    # to help the planner to avoid planning with those wall positions
    parser.add_argument("--use_model_error_tolerated_planner", action="store_true", default=False)

    # for using manual mask
    parser.add_argument("--map_mask_type", type=str, default="manual_global_mask")

    # map filter path
    parser.add_argument("--map_filter_path", type=str, default=None)

    return parser.parse_args()


# main function
if __name__ == '__main__':
    DATA_SPLITS = {
        'split_1': {'trn': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19], 'tst': [0, 1, 2, 3, 14]},
        'split_2': {'trn': [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 18, 19], 'tst': [5, 9, 15, 16, 17]},
        'split_3': {'trn': [0, 1, 2, 3, 5, 7, 9, 10, 11, 14, 15, 16, 17, 19], 'tst': [4, 8, 12, 13, 18]},
        'split_4': {'trn': [0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18], 'tst': [6, 7, 10, 11, 19]},
    }

    # parse the input
    input_args = parse_input()

    # set the randomness
    random.seed(input_args.seed)
    np.random.seed(input_args.seed)
    torch.manual_seed(input_args.seed)

    # set the configurations
    eval_configs = {
        'model_type': input_args.model_type,
        'eval_data': input_args.eval_data,
        'maze_size': input_args.maze_size,
        'max_episode_steps': input_args.max_episode_steps,
        'model_path': input_args.model_path,
        'split_id': input_args.split_id,
        'maze_id': input_args.maze_id,
        'dataset_type': input_args.dataset_type,
        'map_feature': input_args.map_feature,
        'eval_episode_num': input_args.eval_episode_num,
        'device': input_args.device,

        'action': input_args.action,

        'use_model_error_tolerated_planner': input_args.use_model_error_tolerated_planner,

        'map_mask': input_args.map_mask_type,

        'map_filter_path': input_args.map_filter_path
    }

    # create the evaluator
    myEval = HyperModel(eval_configs)

    actions = ['turn_left', 'turn_right', 'forward', 'backward']

    # plot the heat map
    for idx in DATA_SPLITS[f"split_{eval_configs['split_id'] + 1}"][eval_configs['dataset_type']]:
        fig, arr = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f"Action = {actions[input_args.action]}")
        my_map, my_maze, heat_err = myEval.plot_prediction_error_heat_map_ego_motion(idx)

        arr[1, 2].set_title("East")
        h1 = arr[1, 2].imshow(heat_err[0].clip(0, 0.25), cmap='viridis', interpolation='nearest', vmin=0.0, vmax=0.25)
        plt.colorbar(h1, ax=arr[1, 2])
        arr[1, 2].axis('off')

        arr[0, 1].set_title("North")
        h2 = arr[0, 1].imshow(heat_err[1].clip(0, 0.25), cmap='viridis', interpolation='nearest', vmin=0.0, vmax=0.25)
        plt.colorbar(h2, ax=arr[0, 1])
        arr[0, 1].axis('off')

        arr[1, 1].set_title("Map")
        arr[1, 1].imshow(my_maze)
        arr[1, 1].axis('off')

        arr[1, 0].set_title("West")
        h3 = arr[1, 0].imshow(heat_err[2].clip(0, 0.25), cmap='viridis', interpolation='nearest', vmin=0.0, vmax=0.25)
        plt.colorbar(h3, ax=arr[1, 0])
        arr[1, 0].axis('off')

        arr[2, 1].set_title("South")
        h4 = arr[2, 1].imshow(heat_err[3].clip(0, 0.25), cmap='viridis', interpolation='nearest', vmin=0.0, vmax=0.25)
        plt.colorbar(h4, ax=arr[2, 1])
        arr[2, 1].axis('off')

        arr[0, 0].axis('off')
        arr[0, 2].axis('off')
        arr[2, 0].axis('off')
        arr[2, 2].axis('off')

        plt.savefig(f"./plots/maze_{input_args.dataset_type}_{input_args.maze_size}_{idx}_{actions[input_args.action]}_act3.png",
                    dpi=100)

        plt.show()

        print(f"Processing maze {idx}")
