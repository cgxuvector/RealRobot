import numpy as np
from utils.TreeSearchAlg.AStar import A_star
import matplotlib.pyplot as plt
import json


def load_maze(m_size, m_id):
    with open(f"./maze_{m_size}_{m_id}.txt", 'r') as f_in:
        data = f_in.readlines()
    f_in.close()

    data = np.array([[int(float(item)) for item in d.rstrip().split(",")] for d in data])

    data = np.where(data == 0.0, 1.0, data)
    data = np.where(data == 2.0, 0.0, data)

    return data


def find_distances_relationship(m_map):
    results_dict = {}

    [rows, cols] = np.where(m_map == 1.0)
    valid_loc = [(int(r), int(c)) for r, c in zip(rows, cols)]
    loc_num = len(valid_loc)

    for i in range(loc_num-1):
        start_loc = valid_loc[i]
        for j in range(i + 1, loc_num):
            goal_loc = valid_loc[j]
            # compute the distance using A star search
            dist = len(A_star(m_map, start_loc, goal_loc)) - 1
            print(f'start loc = {start_loc}, goal loc = {goal_loc}, distance = {dist}')
            # save the results
            if dist in results_dict.keys():
                results_dict[dist].append([start_loc, goal_loc])
            else:
                results_dict[dist] = [[start_loc, goal_loc]]

    return results_dict


def save(res, m_size, m_id):
    with open(f"./dist_data/maze_{m_size}_{m_id}.json", 'w') as f_out:
        json.dump(res, f_out)
    f_out.close()


if __name__ == "__main__":

    for i in range(20):
        maze_size = 15
        maze_id = i

        maze_map = load_maze(maze_size, maze_id)

        results = find_distances_relationship(maze_map)

        save(results, maze_size, maze_id)




