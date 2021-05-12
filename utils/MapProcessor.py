"""
    This script implements the functions used to process the map
        - From now on, we will use 1 to represent corridors position and 0 for wall positions
"""
import numpy as np
import IPython.terminal.debugger as Debug


def resize(data, target_shape):
    if data.shape[0] == 3:
        # add boundary
        revised_data = np.zeros((5, 5))
        revised_data[1:4, 1:4] = data
        data = revised_data

    # record the source shape
    source_row = data.shape[0] - 2
    source_col = data.shape[1] - 2

    # record the target shape
    target_row = target_shape[0]
    target_col = target_shape[1]

    # compute the scale
    r_scale = target_row // source_row
    c_scale = target_col // source_col

    # scale up the map
    inner_row = source_row * r_scale
    inner_col = source_col * c_scale
    resized_data_inner = np.ones((inner_row, inner_col))
    for r_idx in range(source_row):
        for c_idx in range(source_col):
            s = data[r_idx + 1, c_idx + 1]
            # compute the rect region
            r_idx_from = r_idx * r_scale
            r_idx_to = (r_idx + 1) * r_scale
            c_idx_from = c_idx * c_scale
            c_idx_to = (c_idx + 1) * c_scale
            for c in np.arange(c_idx_from, c_idx_to):
                for r in np.arange(r_idx_from, r_idx_to):
                    if s == 1:
                        resized_data_inner[r, c] = int(1)
                    elif 0 < s < 1:
                        resized_data_inner[r, c] = s
                    else:
                        resized_data_inner[r, c] = int(0)

    # do the padding
    r_lines_start = (target_row - inner_row) // 2
    c_lines_start = (target_col - inner_col) // 2

    # create the final map
    resized_data = np.zeros(target_shape)
    resized_data[r_lines_start:inner_row + r_lines_start, c_lines_start:inner_col + c_lines_start] = resized_data_inner

    return resized_data


def resize_optim(data, target_shape):
    if data.shape[0] == 3:
        # add boundary
        revised_data = np.zeros((5, 5))
        revised_data[1:4, 1:4] = data
        data = revised_data

    # record the source shape
    source_row = data.shape[0] - 2
    source_col = data.shape[1] - 2

    # record the target shape
    target_row = target_shape[0]
    target_col = target_shape[1]

    # compute the scale
    r_scale = target_row // source_row
    c_scale = target_col // source_col

    # scale up the map
    inner_row = source_row * r_scale
    inner_col = source_col * c_scale
    resized_data_inner = np.ones((inner_row, inner_col))

    # scale up the map
    [resized_data_inner[r * r_scale:(r + 1) * r_scale,
     c * c_scale:(c + 1) * c_scale].fill(data[r + 1, c + 1])
     for r in range(source_row) for c in range(source_col)]

    # do the padding
    r_lines_start = (target_row - inner_row) // 2
    c_lines_start = (target_col - inner_col) // 2

    # create the final map
    resized_data = np.zeros(target_shape)
    resized_data[r_lines_start:inner_row + r_lines_start, c_lines_start:inner_col + c_lines_start] = resized_data_inner

    return resized_data


def manual_mask(loc, m):
    # Todo: it is useful to add a scope controller
    loc = loc[0:2] if len(loc) == 3 else loc  # for egocentric action

    # convert back to the map location
    map_loc = (np.array(loc) // 3).astype(int) + np.array([1, 1])

    # compute the rectangle of the local map
    from_loc = map_loc + np.array([-1, -1])
    to_loc = map_loc + np.array([1, 1])

    # create the mask base
    masked_global_map = np.zeros((m.shape[0], m.shape[0]))
    # crop the local map
    local_map = m[from_loc[0]:to_loc[0] + 1, from_loc[1]:to_loc[1] + 1]
    # local_map = np.where(local_map == 0.0, 0.1, local_map)  # change the wall value in the masked area to be 0.1
    # add the local map to the mask base
    masked_global_map[from_loc[0]:to_loc[0] + 1, from_loc[1]:to_loc[1] + 1] = local_map

    return masked_global_map, local_map


def compute_mask(loc, m):
    map_loc = np.array(loc) // 3
    mask = np.zeros_like(m)
    mask[map_loc[0], map_loc[1]] = 1

    return mask
