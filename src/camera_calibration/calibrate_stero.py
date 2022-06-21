#%%
import cv2
import glob
import numpy as np

#%%
def stereo_calibrate(
    clb_folder, clb_dict_loc, stero_dict_loc, image_folder, cam0_loc, cam1_loc
):

    clb_dict = np.load(clb_folder + clb_dict_loc, allow_pickle="TRUE").item()
    mtx1 = clb_dict["c0_mtx"]
    dst1 = clb_dict["c0_dst"]
    mtx2 = clb_dict["c1_mtx"]
    dst2 = clb_dict["c1_dst"]

    c0_images = sorted(glob.glob(image_folder + cam0_loc))
    c1_images = sorted(glob.glob(image_folder + cam1_loc))

    # frame dimensions. Frames should be the same size.
    width = cv2.imread(c0_images[0]).shape[1]
    height = cv2.imread(c0_images[0]).shape[0]

    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 9  # number of checkerboard rows.
    columns = 6  # number of checkerboard columns.
    world_scaling = 1.0  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for fname1, fname2 in zip(c0_images, c1_images):

        frame1 = cv2.imread(fname1, 1)
        frame2 = cv2.imread(fname2, 1)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv2.imshow("img", frame1)

            cv2.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv2.imshow("img2", frame2)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

        cv2.destroyAllWindows()

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

    ret, CM1, s_dist1, CM2, d_dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx1,
        dst1,
        mtx2,
        dst2,
        (width, height),
        criteria=criteria,
        flags=stereocalibration_flags,
    )

    print(ret)
    stero_dict = {"rot_mtx": R, "trn_mtx": T}
    np.save(clb_folder + stero_dict_loc, stero_dict)
    return stero_dict


stero_dict = stereo_calibrate(
    clb_folder="calibration_matrix/",
    clb_dict_loc="calibration_dict.npy",
    stero_dict_loc="stero_clb_dict.npy",
    image_folder="images/",
    cam0_loc="c0_syc/*",
    cam1_loc="c1_syc/*",
)

#%%
def calc_projection_matrix(base_folder, calb_str, stero_str, projection_str):
    clb_dict = np.load(base_folder + calb_str, allow_pickle="TRUE").item()
    stero_dict = np.load(base_folder + stero_str, allow_pickle="TRUE").item()

    mtx1 = clb_dict["c0_mtx"]
    dst1 = clb_dict["c0_dst"]
    mtx2 = clb_dict["c1_mtx"]
    dst2 = clb_dict["c1_dst"]

    R = stero_dict["rot_mtx"]
    T = stero_dict["trn_mtx"]

    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    proj_dict = {"P1": P1, "P2": P2}
    np.save(base_folder + projection_str, proj_dict)

    return proj_dict


proj_dict = calc_projection_matrix(
    base_folder="calibration_matrix/",
    calb_str="calibration_dict.npy",
    stero_str="stero_clb_dict.npy",
    projection_str="projection_dict.npy",
)
# %%
