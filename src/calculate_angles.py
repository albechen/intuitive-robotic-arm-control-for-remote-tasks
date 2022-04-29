#%%
import numpy as np

#%%
def law_of_cos(e1, e2, opp):
    opp_angle = np.arccos((e1 ** 2 + e2 ** 2 - opp ** 2) / (2 * e1 * e2))
    return opp_angle


def calc_raw_arm_angles(END_CORD, lens):
    X, Y, Z = END_CORD[0], END_CORD[1], END_CORD[2]

    theta_1 = np.arcsin(Y / (X ** 2 + Y ** 2) ** 0.5)

    z_adj = Z - lens[0]
    d = sum([X ** 2, Y ** 2, z_adj ** 2]) ** 0.5

    theta_21 = np.arctan(z_adj / (X ** 2 + Y ** 2) ** 0.5)
    theta_22 = law_of_cos(lens[1], d, lens[2] + lens[3])
    theta_2 = theta_21 + theta_22

    theta_3 = np.radians(180) - law_of_cos(lens[1], lens[2] + lens[3], d)

    return [np.degrees(theta_1), np.degrees(theta_2), np.degrees(theta_3)]


def bound_angles(raw_angles_list, bounds_list):
    adj_angles_list = []
    for angle, bound in zip(raw_angles_list, bounds_list):
        if angle <= bound["min"]:
            adj_angle = bound["min"]
        elif angle >= bound["max"]:
            adj_angle = bound["max"]
        else:
            adj_angle = angle
        adj_angles_list.append(round(adj_angle))
    return adj_angles_list


def calc_arm_angles(END_CORD, lens, bounds_list):
    raw_angles_list = calc_raw_arm_angles(END_CORD, lens)
    angles_list = bound_angles(raw_angles_list, bounds_list)
    return angles_list[0], angles_list[1], angles_list[2]