#%%
import cv2
import glob
import numpy as np

# %%
def calibrate_camera(show_image, images_folder):

    images = sorted(glob.glob(images_folder))

    # frame dimensions. Frames should be the same size.
    width = cv2.imread(images[0]).shape[1]
    height = cv2.imread(images[0]).shape[0]

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 9  # number of checkerboard rows.
    columns = 6  # number of checkerboard columns.
    world_scaling = 1.0  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    retList = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
        retList.append(ret)
        print(fname, ret)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            corners = cv2.drawChessboardCorners(img, (rows, columns), corners, ret)
            if show_image == True:
                cv2.imshow(fname, img)
                cv2.waitKey(-1)
            else:
                pass
            cv2.destroyAllWindows()

    ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )

    nmtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dst, (width, height), 1, (width, height)
    )
    print("rmse:", ret)
    print("camera matrix:\n", mtx)
    print("distortion coeffs:", dst)

    return mtx, dst, retList, nmtx, roi


#%%
def list_frames_no_board(cam0, cam1):
    mtx0, dst0, retList0, nmtx0, roi0 = calibrate_camera(
        False, images_folder="images/sync/%s/*" % cam0
    )
    mtx1, dst1, retList1, nmtx1, roi1 = calibrate_camera(
        False, images_folder="images/sync/%s/*" % cam1
    )

    false_list = [f0 == f1 for f0, f1 in zip(retList0, retList1)]
    for flag, fn0, fn1 in zip(
        false_list,
        sorted(glob.glob("images/sync/%s/*" % cam0)),
        sorted(glob.glob("images/sync/%s/*" % cam1)),
    ):
        if flag == False:
            print(fn0, fn1)


def show_each_image(cam_list):
    for cam in cam_list:
        mtx, dst, retList, nmtx, roi = calibrate_camera(
            True, images_folder="images/sync/%s/*" % cam
        )


def finalize_calibration(cam_list):
    calibration_dict = {}
    for cam in cam_list:
        mtx, dst, retList, nmtx, roi = calibrate_camera(
            False, images_folder="images/sync/%s/*" % cam
        )

        calibration_dict[cam[:2]] = {"mtx": mtx, "dst": dst, "nmtx": nmtx, "roi": roi}

    np.save("calibration_matrix/calibration_dict.npy", calibration_dict)


# %%
# list_frames_no_board("c0", "c1")

# #%%
# show_each_image(["c0", "c1"])

# #%%
finalize_calibration(["c0", "c1"])

# %%
# clb_dict = np.load(
#     "calibration_matrix/calibration_dict.npy", allow_pickle="TRUE"
# ).item()
# clb_dict
# #%%
# cam = 'c1'

# img = cv2.imread('images/sync/c0/20220603_134410.png'.format(cam))
# cam_dict = clb_dict[cam]

# dst = cv2.undistort(
#     img, cam_dict["mtx"], cam_dict["dst"], None, cam_dict["nmtx"]
# )

# x, y, w, h = cam_dict["roi"]
# # dst = dst[y : y + h, x : x + w]
# print(dst.shape, img.shape)
# cv2.imshow("undistort", dst)
# cv2.imshow("img", img)
# cv2.waitKey(-1)
# cv2.destroyAllWindows()

# %%
