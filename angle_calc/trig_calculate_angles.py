#%%
import math
import numpy as np

start_loc = np.array([0, 0, 0])
end_loc = np.array([5, 5, 5])

MIN_ROTATION = 0
MAX_ROTATION = 180
LIMB_A_LEN = 4
LIMB_B_LEN = 5


def array_xyz(array):
    x, y, z = array[0], array[1], array[2]
    return x, y, z


def distance_two_points(start_loc, end_loc):
    vector = end_loc - start_loc
    dist = np.dot(vector, vector.T) ** 0.5
    return dist


def distance_in_range_check(start_loc, end_loc):
    dist = distance_two_points(start_loc, end_loc)
    max_dist = LIMB_A_LEN + LIMB_B_LEN
    min_dist = abs(LIMB_A_LEN - LIMB_B_LEN)
    print(dist)
    if max_dist < dist:
        print("MAX: NOT IN RANGE")
    elif min_dist > dist:
        print("MIN: NOT IN RANGE")
    else:
        print("GOOD TO GO")


distance_in_range_check(start_loc, end_loc)

# %%
def calc_sss_angle(c1, c2, f1):
    angle_rad = math.acos((c1 ** 2 + c2 ** 2 - f1 ** 2) / (2 * c1 * c2))
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def calc_aas_sides(angle, side_op_90):
    last_angle = 180 - angle - 90
    side_opp_angle = (
        side_op_90 * math.sin(math.radians(angle)) / math.sin(math.radians(90))
    )
    side_between_angles = (
        side_op_90 * math.sin(math.radians(last_angle)) / math.sin(math.radians(90))
    )
    return side_opp_angle, side_between_angles


def larger_limb_points(start_loc, end_loc):
    # MOTOR 0
    vector = end_loc - start_loc
    x, y, z = array_xyz(vector)
    xy = (x ** 2 + y ** 2) ** 0.5
    m0_agl = calc_sss_angle(x, xy, y)
    m0_pt = start_loc

    # MOTOR 1
    dist = distance_two_points(start_loc, end_loc)
    s_AB = LIMB_A_LEN
    s_BC = LIMB_B_LEN
    s_AC = dist
    s_CD = z
    s_AD = xy

    a_BAC = calc_sss_angle(s_AC, s_AB, s_BC)
    a_ABC = calc_sss_angle(s_AB, s_BC, s_AC)
    a_ACB = calc_sss_angle(s_BC, s_AC, s_AB)

    a_DAC = calc_sss_angle(s_AD, s_AC, s_CD)
    a_ACD = calc_sss_angle(s_AC, s_CD, s_AD)
    a_CDA = calc_sss_angle(s_CD, s_AD, s_AC)

    m1_agl = a_DAC + a_BAC
    m1_pt = start_loc

    # MOTOR 2
    if m1_agl < 90:
        a_temp = m1_agl
        m2_z, m2_xy = calc_aas_sides(a_temp, s_AB)
    else:
        a_temp = 180 - m1_agl
        m2_z, m2_xy = calc_aas_sides(a_temp, s_AB)
        m2_xy = -m2_xy

    m2_agl = a_ABC
    m2_y, m2_x = calc_aas_sides(m0_agl, m2_xy)
    m2_pt = np.array([m2_x, m2_y, m2_z])

    print(distance_two_points(start_loc, m2_pt), distance_two_points(m2_pt, end_loc))

    return (m0_agl, m0_pt, m1_agl, m1_pt, m2_agl, m2_pt)


#%%
larger_limb_points(start_loc, end_loc)