#%%
import cv2 as cv
import glob
import numpy as np
from scipy import linalg
import os
import pickle

#%%
# set path to current dierectory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}


def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def replace_pickle(path, newFile):
    with open(path, "wb") as f:
        pickle.dump(newFile, f)


# Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3, 0:3] / Vh[3, 3]


# Open and load the calibration_settings.yaml file
# def parse_calibration_settings_file(filename):

#     global calibration_settings

#     if not os.path.exists(filename):
#         print('File does not exist:', filename)
#         quit()

#     print('Using for calibration settings: ', filename)

#     with open(filename) as f:
#         calibration_settings = yaml.safe_load(f)

#     #rudimentray check to make sure correct file was loaded
#     if 'camera0' not in calibration_settings.keys():
#         print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
#         quit()

calibration_settings = {
    "camera0": 0,
    "camera1": 1,
    "frame_width": 1920,
    "frame_height": 1080,
    "mono_calibration_frames": 10,
    "stereo_calibration_frames": 10,
    "view_resize": 1,
    "checkerboard_box_size_scale": 2.3,
    "checkerboard_rows": 6,
    "checkerboard_columns": 9,
    "cooldown": 100,
}


# Open camera stream and save frames
def save_frames_single_camera(camera_name):

    # get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    number_to_save = calibration_settings["mono_calibration_frames"]
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]

    # open video stream and change resolution.
    # Note: if unsupported resolution is used, this does NOT raise an error.
    cap = cv.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:

        ret, frame = cap.read()
        if ret == False:
            # if no video data is received, can't calibrate the camera, so exit.
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv.resize(frame, None, fx=1 / view_resize, fy=1 / view_resize)

        if not start:
            cv.putText(
                frame_small,
                "Press SPACEBAR to start collection frames",
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

        if start:
            cooldown -= 1
            cv.putText(
                frame_small,
                "Cooldown: " + str(cooldown),
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame_small,
                "Num frames: " + str(saved_count),
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            # save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = "frames/" + camera_name + "_" + str(saved_count) + ".png"
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow("frame_small", frame_small)
        k = cv.waitKey(1)

        if k == 27:
            # if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            # Press spacebar to start data collection
            start = True

        # break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()


# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):

    # NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

    # read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings[
        "checkerboard_box_size_scale"
    ]  # this will change to user defined length scale

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.putText(
                frame,
                'If detected points are poor, press "s" to skip this sample',
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

            cv.imshow("img", frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord("s"):
                print("skipping")
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )
    print("rmse:", ret)
    print("camera matrix:\n", cmtx)
    print("distortion coeffs:", dist)

    return cmtx, dist


# open both cameras and take calibration frames
def save_frames_two_cams(
    camera0_name,
    camera1_name,
    prefix="",
    number_to_save=calibration_settings["stereo_calibration_frames"],
):

    # settings for taking data
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]

    # open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    # set camera resolutions
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Cameras not returning video data. Exiting...")
            quit()

        frame0_small = cv.resize(
            frame0, None, fx=1.0 / view_resize, fy=1.0 / view_resize
        )
        frame1_small = cv.resize(
            frame1, None, fx=1.0 / view_resize, fy=1.0 / view_resize
        )

        if not start:
            cv.putText(
                frame0_small,
                "Make sure both cameras can see the calibration pattern well",
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.putText(
                frame0_small,
                "Press SPACEBAR to start collection frames",
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

        if start:
            cooldown -= 1
            cv.putText(
                frame0_small,
                "Cooldown: " + str(cooldown),
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame0_small,
                "Num frames: " + str(saved_count),
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            cv.putText(
                frame1_small,
                "Cooldown: " + str(cooldown),
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame1_small,
                "Num frames: " + str(saved_count),
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            # save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = (
                    "frames_pair/"
                    + camera0_name
                    + "_"
                    + prefix
                    + str(saved_count)
                    + ".png"
                )
                cv.imwrite(savename, frame0)

                savename = (
                    "frames_pair/"
                    + camera1_name
                    + "_"
                    + prefix
                    + str(saved_count)
                    + ".png"
                )
                cv.imwrite(savename, frame1)

                saved_count += 1
                cooldown = cooldown_time

        cv.imshow("frame0_small", frame0_small)
        cv.imshow("frame1_small", frame1_small)
        k = cv.waitKey(1)

        if k == 27:
            # if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            # Press spacebar to start data collection
            start = True

        # break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()


# open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    # read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    # open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # calibration pattern settings
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0, 0].astype(np.int32)
            p0_c2 = corners2[0, 0].astype(np.int32)

            cv.putText(
                frame0,
                "O",
                (p0_c1[0], p0_c1[1]),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv.imshow("img", frame0)

            cv.putText(
                frame1,
                "O",
                (p0_c2[0], p0_c2[1]),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)
            cv.imshow("img2", frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord("s"):
                print("skipping")
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx0,
        dist0,
        mtx1,
        dist1,
        (width, height),
        criteria=criteria,
        flags=stereocalibration_flags,
    )

    print("rmse: ", ret)
    cv.destroyAllWindows()
    return R, T


# Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1

    return P


# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P


# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(
    camera0_name, camera0_data, camera1_name, camera1_data, _zshift=50.0
):

    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    # define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    z_shift = np.array([0.0, 0.0, _zshift]).reshape((1, 3))
    # increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    # project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    # Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.0])

        # project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera0.append(uv)

        # project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera1.append(uv)

    # these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    # open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    # set camera resolutions
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Video stream not returning frame data")
            quit()

        # follow RGB colors to indicate XYZ axes respectively
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        # draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)

        # draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow("frame0", frame0)
        cv.imshow("frame1", frame1)

        k = cv.waitKey(1)
        if k == 27:
            break

    cv.destroyAllWindows()


def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    # calibration pattern settings
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
    cv.putText(
        frame,
        "If you don't see detected points, try with a different image",
        (50, 50),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        1,
    )
    cv.imshow("img", frame)
    k = cv.waitKey(-1)
    if k == 27:
        cv.destroyAllWindows()

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _ = cv.Rodrigues(rvec)  # rvec is Rotation matrix in Rodrigues vector form

    return R, tvec


def get_cam1_to_world_transforms(
    cmtx0, dist0, R_W0, T_W0, cmtx1, dist1, R_01, T_01, image_path0, image_path1
):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32"
    ).reshape((4, 1, 3))
    # axes colors are RGB format to indicate XYZ axes.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    # project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow("frame0", frame0)
    cv.imshow("frame1", frame1)
    k = cv.waitKey(-1)
    if k == 27:
        cv.destroyAllWindows()

    return R_W1, T_W1


#%%
"""Step1. Save calibration frames for single cameras"""
save_frames_single_camera("camera0")  # save frames for camera0
save_frames_single_camera("camera1")  # save frames for camera1

#%%
"""Step2. Obtain camera intrinsic matrices and save them"""
# camera0 intrinsics
images_prefix = "frames/camera0*"
cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix)
images_prefix = "frames/camera1*"
cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)

#%%
"""Step3. Save calibration frames for both cameras simultaneously"""
save_frames_two_cams("camera0", "camera1")  # save simultaneous frames

#%%
"""Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
frames_prefix_c0 = "frames_pair/camera0*"
frames_prefix_c1 = "frames_pair/camera1*"
R1, T1 = stereo_calibrate(
    cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1
)

R0 = np.eye(3, dtype=np.float32)
T0 = np.array([0.0, 0.0, 0.0]).reshape((3, 1))

P0 = get_projection_matrix(cmtx0, R0, T0)
P1 = get_projection_matrix(cmtx1, R1, T1)

#%%
"""Step5. Save calibration data where camera0 defines the world space origin."""
# camera0 rotation and translation is identity matrix and zeros vector
# check your calibration makes sense
camera0_data = [cmtx0, dist0, R0, T0]
camera1_data = [cmtx1, dist1, R1, T1]
check_calibration("camera0", camera0_data, "camera1", camera1_data, _zshift=60.0)

#%%
# """Optional. Define a different origin point and save the calibration data"""
# save_frames_two_cams("camera0", "camera1", "world", 1)

# #%%
# # get the world to camera0 rotation and translation
# R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, "frames_pair/camera0_world0.png")

# #%%
# # get rotation and translation from world directly to camera1
# R_W1, T_W1 = get_cam1_to_world_transforms(
#     cmtx0,
#     dist0,
#     R_W0,
#     T_W0,
#     cmtx1,
#     dist1,
#     R1,
#     T1,
#     "frames_pair/camera0_world0.png",
#     "frames_pair/camera1_world0.png",
# )

# P_W0 = get_projection_matrix(cmtx0, R_W0, T_W0)
# P_W1 = get_projection_matrix(cmtx1, R_W1, T_W1)

# #%%
# camera0_data = [cmtx0, dist0, R_W0, T_W0]
# camera1_data = [cmtx1, dist1, R_W1, T_W1]
# check_calibration("camera0", camera0_data, "camera1", camera1_data, _zshift=0)

#%%
param_dict = {
    "cam0": {
        "mtx": cmtx0,
        "dist": dist0,
        "R": R0,
        "T": T0,
        "P": P0,
    },
    "cam1": {
        "mtx": cmtx1,
        "dist": dist1,
        "R": R1,
        "T": T1,
        "P": P1,
    },
}
replace_pickle("camera_parameters/cam_params.pkl", param_dict)
# %%
