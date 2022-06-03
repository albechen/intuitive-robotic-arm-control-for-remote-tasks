#%%
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg

#%%
def get_c0_c1_points(cal_folder, ann_pts_str, img_num):
    with open(cal_folder + ann_pts_str, "rb") as handle:
        all_points = pickle.load(handle)

    img = "pic_" + str(img_num)
    c0_pt = all_points["c0"][img]["left"] + all_points["c0"][img]["right"]
    c1_pt = all_points["c1"][img]["left"] + all_points["c1"][img]["right"]

    return np.array(c0_pt), np.array(c1_pt)


def load_all_points_for_camera(image_folder, img_num, c0_pt, c1_pt):

    c0_files = sorted(glob.glob(image_folder + "c0_hand/*"))
    c1_files = sorted(glob.glob(image_folder + "c1_hand/*"))

    frame1 = cv2.imread(c0_files[img_num])
    frame2 = cv2.imread(c1_files[img_num])

    plt.imshow(frame1[:, :, [2, 1, 0]])
    plt.scatter(c0_pt[:, 0], c0_pt[:, 1])
    plt.show()

    plt.imshow(frame2[:, :, [2, 1, 0]])
    plt.scatter(c1_pt[:, 0], c1_pt[:, 1])
    plt.show()


def DLT(cal_folder, proj_loc, point1, point2):
    proj_dict = np.load(cal_folder + proj_loc, allow_pickle="TRUE").item()
    P1 = proj_dict["P1"]
    P2 = proj_dict["P2"]

    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    # print("Triangulated point: ")
    # print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]


def show_annotated_image(img_num, img_folder, cal_folder, ann_pts_loc):
    c0_pt, c1_pt = get_c0_c1_points(cal_folder, ann_pts_loc, img_num)
    load_all_points_for_camera(img_folder, img_num, c0_pt, c1_pt)


def calc_adjusted_3d_points(img_num, cal_folder, ann_pts_loc, proj_loc):
    c0_pt, c1_pt = get_c0_c1_points(cal_folder, ann_pts_loc, img_num)
    p3ds = []
    for uv1, uv2 in zip(c0_pt, c1_pt):
        _p3d = DLT(cal_folder, proj_loc, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    return p3ds


def graph_3d_space(img_num, cal_folder, ann_pts_loc, proj_loc):
    p3ds = calc_adjusted_3d_points(img_num, cal_folder, ann_pts_loc, proj_loc)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d(-15, 5)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(10, 30)

    conn_left = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
    ]

    conn_right = [[y + 13 for y in x] for x in conn_left]
    conn = conn_left + conn_right

    for _c in conn:
        # print(p3ds[_c[0]])
        # print(p3ds[_c[1]])
        ax.plot(
            xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]],
            ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
            zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]],
            c="red",
        )

    plt.show()


#%%
cal_folder = "calibration_matrix/"
proj_loc = "projection_dict.npy"
ann_pts_loc = "annotated_points.pkl"
img_folder = "calibration_images/"
img_num = 1

show_annotated_image(img_num, img_folder, cal_folder, ann_pts_loc)
graph_3d_space(img_num, cal_folder, ann_pts_loc, proj_loc)

#%%