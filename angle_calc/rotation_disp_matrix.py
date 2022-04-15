#%%
import numpy as np

anlges = [0, 0, 0, 0, 0, 0]
angles_rad_list = [np.radians(x) for x in anlges]

lens = [1, 5, 7, 1, 1, 2]

rot_matrix_list = [
    np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
]

disp_matrix_list = [
    np.array([[0], [0], [lens[0]]]),
    np.array(
        [
            [lens[1] * np.cos(angles_rad_list[1])],
            [lens[1] * np.sin(angles_rad_list[1])],
            [0],
        ]
    ),
    np.array([[0], [0], [0]]),
    np.array([[0], [0], [lens[2] + lens[3]]]),
    np.array([[0], [0], [0]]),
    np.array([[0], [0], [lens[4] + lens[5]]]),
]


def calc_single_rotation(angle, rot_matrix):
    z_rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rot_matrix = np.dot(z_rotation_matrix, rot_matrix)
    return rot_matrix


def calc_series_h_transforms(angles_rad_list, rot_matrix_list, disp_matrix_list):

    num_joints = len(angles_rad_list)
    homo_matrix_list = [0] * num_joints

    for n in range(num_joints):
        rot_matrix = calc_single_rotation(angles_rad_list[n], rot_matrix_list[n])
        disp_matrix = disp_matrix_list[n]
        homo_matrix = np.concatenate((rot_matrix, disp_matrix), 1)
        homo_matrix = np.concatenate((homo_matrix, [[0, 0, 0, 1]]), 0)
        homo_matrix_list[n] = homo_matrix

    return homo_matrix_list


calc_series_h_transforms(angles_rad_list, rot_matrix_list, disp_matrix_list)