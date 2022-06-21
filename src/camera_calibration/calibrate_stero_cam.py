#%%
import numpy as np
import cv2 as cv
import glob


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9, 6)
frameSize = (1920, 1080)
show_image = False

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.


imagesLeft = glob.glob("images/c1_syc/*")
imagesRight = glob.glob("images/c0_syc/*")

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow("img left", imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow("img right", imgR)
        if show_image == True:
            cv.waitKey(-1)
        else:
            pass


cv.destroyAllWindows()

############## CALIBRATION #######################################################

retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(
    objpoints, imgpointsL, frameSize, None, None
)
heightL, widthL, channelsL = imgL.shape
n_mtxL, roi_L = cv.getOptimalNewCameraMatrix(
    mtxL, distL, (widthL, heightL), 1, (widthL, heightL)
)

retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(
    objpoints, imgpointsR, frameSize, None, None
)
heightR, widthR, channelsR = imgR.shape
n_mtxR, roi_R = cv.getOptimalNewCameraMatrix(
    mtxR, distR, (widthR, heightR), 1, (widthR, heightR)
)
print("Cam 0: ", retR)
print("Cam 1: ", retL)


########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
(ret, CM1, s_dist1, CM2, d_dist2, R, T, E, F) = cv.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL,
    distL,
    mtxR,
    distR,
    grayL.shape[::-1],
    criteria_stereo,
    flags,
)

calibration_dict = {
    "c0_mtx": mtxR,
    "c0_dst": distR,
    "c1_mtx": mtxL,
    "c1_dst": distL,
}

stero_dict = {"rot_mtx": R, "trn_mtx": T}


clb_folder = "calibration_matrix/"
clb_dict_loc = "calibration_dict.npy"
stero_dict_loc = "stero_clb_dict.npy"

np.save(clb_folder + clb_dict_loc, calibration_dict)
np.save(clb_folder + stero_dict_loc, stero_dict)
print("Stero: ", ret)


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