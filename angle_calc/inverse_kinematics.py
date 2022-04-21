#%%
import numpy as np

#%%
def law_of_cos(e1, e2, opp):
    opp_angle = np.arccos((e1 ** 2 + e2 ** 2 - opp ** 2) / (2 * e1 * e2))
    return opp_angle


def calc_angles_of_arm(END_CORD, lens):
    X, Y, Z = END_CORD[0], END_CORD[1], END_CORD[2]

    theta_1 = np.arcsin(Y / (X ** 2 + Y ** 2) ** 0.5)

    z_adj = Z - lens[0]
    d = sum([X ** 2, Y ** 2, z_adj ** 2]) ** 0.5

    theta_21 = np.arctan(z_adj / (X ** 2 + Y ** 2) ** 0.5)
    theta_22 = law_of_cos(lens[1], d, lens[2] + lens[3])
    theta_2 = theta_21 + theta_22

    theta_3 = np.radians(180) - law_of_cos(lens[1], lens[2] + lens[3], d)

    return np.degrees(theta_1), np.degrees(theta_2), np.degrees(theta_3)


def calc_series_rotation(homo_matrix_list, first_value, last_value):
    first_matrix = True
    homo_matrix_slice = homo_matrix_list[first_value:last_value]

    for homo_matrix in homo_matrix_slice:
        if first_matrix == True:
            overall_rot_matrix = homo_matrix
            first_matrix = False
        else:
            overall_rot_matrix = np.dot(overall_rot_matrix, homo_matrix)

    return overall_rot_matrix


def calc_homo_matrix(angles, DH_table):

    num_joints = len(angles)
    homo_matrix_list = [0] * num_joints

    for n in range(num_joints):
        theta, alpha, r, d = DH_table[n]
        theta = np.radians(theta)
        alpha = np.radians(alpha)

        c_theta = round(np.cos(theta), 10)
        s_theta = round(np.sin(theta), 10)
        c_alpha = round(np.cos(alpha), 10)
        s_alpha = round(np.sin(alpha), 10)

        homo_matrix = [
            [
                c_theta,
                -s_theta * c_alpha,
                s_theta * s_alpha,
                r * c_theta,
            ],
            [
                s_theta,
                c_theta * c_alpha,
                -c_theta * s_alpha,
                r * s_theta,
            ],
            [0, s_alpha, c_alpha, d],
            [0, 0, 0, 1],
        ]
        homo_matrix_list[n] = np.array(homo_matrix)

    return homo_matrix_list


def return_dh_table(angles, lens):
    DH_table = [
        [angles[0] + 0, 90, 0, lens[0]],
        [angles[1] + 0, 180, lens[1], 0],
        [angles[2] - 90, -90, 0, 0],
        [angles[3] + 0, -90, 0, lens[2] + lens[3]],
        [angles[4] - 90, 90, 0, 0],
        [angles[5] + 0, 0, 0, lens[4] + lens[5]],
    ]
    return DH_table


def calc_wirst_rot_matrix(wrist_angles):
    wrist_angles_rad = [np.radians(x) for x in wrist_angles]
    agl_z1, agl_y, agl_z12 = wrist_angles_rad

    rot_matrix = np.array(
        [
            [
                np.cos(agl_z1) * np.cos(agl_y) * np.cos(agl_z12)
                - np.sin(agl_z1) * np.sin(agl_z12),
                -np.cos(agl_z1) * np.cos(agl_y) * np.sin(agl_z12)
                - np.sin(agl_z1) * np.cos(agl_z12),
                np.cos(agl_z1) * np.sin(agl_y),
            ],
            [
                np.sin(agl_z1) * np.cos(agl_y) * np.cos(agl_z12)
                + np.cos(agl_z1) * np.sin(agl_z12),
                -np.sin(agl_z1) * np.cos(agl_y) * np.sin(agl_z12)
                + np.cos(agl_z1) * np.cos(agl_z12),
                np.sin(agl_z1) * np.sin(agl_y),
            ],
            [
                -np.sin(agl_y) * np.cos(agl_z12),
                np.sin(agl_y) * np.sin(agl_z12),
                np.cos(agl_y),
            ],
        ]
    )
    # rot_z = np.array(
    #     [
    #         [np.cos(agl_z1), -np.sin(agl_z1), 0],
    #         [np.sin(agl_z1), np.cos(agl_z1), 0],
    #         [0, 0, 1],
    #     ]
    # )
    # rot_y = np.array(
    #     [
    #         [np.cos(agl_y), 0, np.sin(agl_y)],
    #         [0, 1, 0],
    #         [-np.sin(agl_y), 0, np.cos(agl_y)],
    #     ]
    # )
    # rot_x = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, np.cos(agl_z12), -np.sin(agl_z12)],
    #         [0, np.sin(agl_z12), np.cos(agl_z12)],
    #     ]
    # )
    # rot_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)
    return rot_matrix


#%%
END_CORD = [0, 8, 6]
lens = [1, 5, 7, 1, 1, 2]

calc_angles_of_arm(END_CORD, lens)

#%%
angles = [90, 90, 90, 0, 90, 0]
DH_table = return_dh_table(angles, lens)

homo_matrix_list = calc_homo_matrix(angles, DH_table)

calc_series_rotation(homo_matrix_list, 0, 6)
# np.dot(calc_series_rotation(homo_matrix_list, 0, 6)[:3, :3], np.array([[1], [0], [0]]))

#%%
angles = [90, 90, 90, 0, 0, 0]
DH_table = return_dh_table(angles, lens)

homo_matrix_list = calc_homo_matrix(angles, DH_table)

rot_03 = calc_series_rotation(homo_matrix_list, 0, 3)[:3, :3]
inv_rot_03 = np.linalg.inv(rot_03)
rot_06 = [
    [0, -1, 0],
    [0, 0, 1],
    [-1, 0, 0],
]

rot_36 = np.dot(inv_rot_03, rot_06)
theta_5 = np.arcsin(rot_36[2, 2])
if theta_5 != 0:
    if rot_36[0, 2] == 0:
        theta_6 = np.pi / 2
    else:
        theta_6 = np.arctan(rot_36[1, 2] / rot_36[0, 2])
    if rot_36[2, 0] == 0:
        theta_4 = np.pi / 2
    else:
        theta_4 = np.arctan(-rot_36[2, 1] / rot_36[2, 0])
else:
    print("RIP")

print(np.degrees(theta_4), np.degrees(theta_5), np.degrees(theta_6))

#%%

wrist_angles = [0, 180, 90]
rot_matrix = calc_wirst_rot_matrix(wrist_angles)
np.dot(rot_matrix, np.array([[0], [0], [1]]))

# %%
middle_joint = np.array([6, 7, 7])
ring_joint = np.array([8, 7, 8])
end_hand_loc = (middle_joint + ring_joint) / 2


#%%
a = np.array([0, 0, 1])
b = np.array([1, 1, 1])


def align_vectors(a, b):
    b = b / np.linalg.norm(b)
    a = a / np.linalg.norm(a)
    v = np.cross(a, b)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R


def angle(v1, v2):
    radians = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    return radians


# def angle(a, b):
#     """Angle between vectors"""
#     a = a / np.linalg.norm(a)
#     b = b / np.linalg.norm(b)
#     return np.arccos(a.dot(b))

x_axis = [1, 0, 0]
test = angle(x_axis, [1, -1, 0])
np.degrees(test)


#%%
point = np.array([-0.02, 1.004, -0.02])
direction = np.array([1.0, 0.0, 0.0])
rotation = align_vectors(point, direction)

# Rotate point in align with direction. The result vector is aligned with direction
result = rotation.dot(point)
print(result)
print("Angle:", angle(direction, point))  # 0.0
print("Length:", np.isclose(np.linalg.norm(point), np.linalg.norm(result)))  # True


# Rotate direction by the matrix, result does not align with direction but the angle between the original vector (direction) and the result2 are the same.
result2 = rotation.dot(direction)
print(result2)
print(
    "Same Angle:", np.isclose(angle(point, result), angle(direction, result2))
)  # True
print("Length:", np.isclose(np.linalg.norm(direction), np.linalg.norm(result2)))  # True
