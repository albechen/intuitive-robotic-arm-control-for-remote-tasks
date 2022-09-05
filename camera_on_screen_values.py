#%%
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
from src.camera_calibration.utils import DLT
from src.angle_calc.inverse_kinematics import calculate_angles_given_joint_loc

import serial
import struct

arduino_serial = serial.Serial("COM5", 9600)

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


def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


#%%

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

frame_shape = [1080, 1920]

# add here if you need more keypoints
# "WRIST",
# "THUMB_CMC",
# "THUMB_MCP",
# "THUMB_IP",
# "THUMB_TIP",
# "INDEX_FINGER_MCP",
# "INDEX_FINGER_PIP",
# "INDEX_FINGER_DIP",
# "INDEX_FINGER_TIP",
# "RING_FINGER_MCP",
# "RING_FINGER_PIP",
# "RING_FINGER_DIP",
# "RING_FINGER_TIP",

joint_list = [
    "WRIST",
    "THUMB_TIP",
    "INDEX_FINGER_TIP",
    "PINKY_TIP",
]

bounds_list = [
    {"min": 0, "max": 180},
    {"min": 0, "max": 135},
    {"min": 0, "max": 135},
    {"min": 0, "max": 180},
    {"min": 0, "max": 180},
    {"min": 0, "max": 180},
]
lens = [8, 24.5, 23, 3, 5, 2]


def extract_x_y_z_cords(ih, iw, cords):
    return [int(round(cords.x * iw)), int(round(cords.y * ih))]


def get_joint_cords(results, ih, iw, joint_name):
    right_joint_cord = [0, 0]
    left_joint_cord = [0, 0]

    for handness, hand_landmarks in zip(
        results.multi_handedness, results.multi_hand_landmarks
    ):
        joint_landmark = hand_landmarks.landmark[mp_hands.HandLandmark[joint_name]]
        joint_cords = extract_x_y_z_cords(ih, iw, joint_landmark)

        if handness.classification[0].label == "Left":
            right_joint_cord = joint_cords
        elif handness.classification[0].label == "Right":
            left_joint_cord = joint_cords

    return right_joint_cord, left_joint_cord


############################ CODE TO DRAW ORIGIN LINES ############################
def gather_2d_points_given_3d(P0, P1, draw_axes_points):
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

    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    return pixel_points_camera0, pixel_points_camera1


def offset_3d_origin(pt, new_origin):
    kp = np.array(pt) - np.array(new_origin)
    kp_adj = [-kp[0], -kp[2], -kp[1]]
    kp_adj = [round(x) for x in kp_adj]
    return kp_adj


def check_calibration(P0, P1, left_origin, right_origin):
    # define coordinate axes in 3D space. These are just the usual coorindate vectors
    # increase the size of the coorindate axes
    coordinate_points = np.array(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]
    )

    # project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    # Note that this uses homogenous coordinate formulation
    left_shift = np.array(left_origin).reshape((1, 3))
    left_pts = 5 * coordinate_points + left_shift
    left_cam0, left_cam1 = gather_2d_points_given_3d(P0, P1, left_pts)

    right_shift = np.array(right_origin).reshape((1, 3))
    right_pts = 5 * coordinate_points + right_shift
    right_cam0, right_cam1 = gather_2d_points_given_3d(P0, P1, right_pts)

    return left_cam0, left_cam1, right_cam0, right_cam1


def draw_origin_lines(frame0, frame1, cam0_pts, cam1_pts):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    # draw projections to camera0
    origin = tuple(cam0_pts[0].astype(np.int32))
    for col, _p in zip(colors, cam0_pts[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    # draw projections to camera1
    origin = tuple(cam1_pts[0].astype(np.int32))
    for col, _p in zip(colors, cam1_pts[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)


########################### END ###########################

###### MAIN BODY SHOWING WRITTEN ON VIDEO ###################
def run_mp(input_stream1, input_stream2, P0, P1):
    # input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    # create body keypoints detector objects.
    hand_0 = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
    )
    hand_1 = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
    )

    left_origin = [25, 20, 70]
    right_origin = [-25, 20, 70]
    left_cam0, left_cam1, right_cam0, right_cam1 = check_calibration(
        P0, P1, left_origin, right_origin
    )

    # containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into memory issue if you run the program without stop
    # kpts_cam0 = []
    # kpts_cam1 = []
    # kpts_3d = []

    # initialize avg list for angles
    angles_list = [[0, 0, 0, 0, 0, 0, 0]]
    num_to_avg = 10

    while 1:

        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # crop to 720x720.
        # Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        # if frame0.shape[1] != 720:
        #     frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
        #     frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = hand_0.process(frame0)
        results1 = hand_1.process(frame1)

        # reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        width = frame0.shape[1]
        height = frame0.shape[0]

        draw_origin_lines(frame0, frame1, left_cam0, left_cam1)
        draw_origin_lines(frame0, frame1, right_cam0, right_cam1)

        # check for keypoints detection
        frame_dict = {
            "cam_0": {"left": {}, "right": {}},
            "cam_1": {"left": {}, "right": {}},
        }
        if results0.multi_hand_landmarks:
            for joint in joint_list:
                right, left = get_joint_cords(results0, height, width, joint)
                cv.circle(frame0, right, 3, (0, 0, 255), -1)
                cv.circle(frame0, left, 3, (0, 0, 255), -1)
                frame_dict["cam_0"]["right"][joint] = right
                frame_dict["cam_0"]["left"][joint] = left
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            for joint in joint_list:
                frame_dict["cam_0"]["right"][joint] = [0, 0]
                frame_dict["cam_0"]["left"][joint] = [0, 0]

        if results1.multi_hand_landmarks:
            for joint in joint_list:
                right, left = get_joint_cords(results1, height, width, joint)
                cv.circle(frame1, right, 3, (0, 0, 255), -1)
                cv.circle(frame1, left, 3, (0, 0, 255), -1)
                frame_dict["cam_1"]["right"][joint] = right
                frame_dict["cam_1"]["left"][joint] = left
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            for joint in joint_list:
                frame_dict["cam_1"]["right"][joint] = [0, 0]
                frame_dict["cam_1"]["left"][joint] = [0, 0]

        # calculate 3d position
        kp_3d = {"left": {}, "right": {}}
        angles_dict = {"left": {}, "right": {}}

        for dir in ["left", "right"]:
            if dir == "left":
                new_origin = left_origin
            elif dir == "right":
                new_origin = right_origin

            for joint in joint_list:

                kp_0 = frame_dict["cam_0"][dir][joint]
                kp_1 = frame_dict["cam_1"][dir][joint]

                if kp_0 == [0, 0] or kp_1 == [0, 0]:
                    kp_3d[dir][joint] = [0, 0, 0]
                    kp_3d[dir]["raw_" + joint] = [0, 0, 0]

                else:
                    kp = DLT(P0, P1, kp_0, kp_1)
                    kp_3d[dir]["raw_" + joint] = [round(x) for x in kp]
                    kp_3d[dir][joint] = offset_3d_origin(kp, new_origin)

            if kp_3d[dir]["WRIST"] == [0, 0, 0]:
                adj_angles = [0, 0, 0, 0, 0, 0]
                claw_agl = 255
            else:
                (
                    raw_angles,
                    adj_angles,
                    end_rot_matrix,
                    claw_agl,
                ) = calculate_angles_given_joint_loc(
                    kp_3d[dir]["WRIST"],
                    kp_3d[dir]["PINKY_TIP"],
                    kp_3d[dir]["INDEX_FINGER_TIP"],
                    kp_3d[dir]["THUMB_TIP"],
                    lens,
                    bounds_list,
                )
            angles = adj_angles + [claw_agl]
            angles_dict[dir]["angles"] = angles

            # AVG OUT ANGLES AND SAVE
            # catch instances where failed angle calc
            # average out certain number of angles and remove latest
            # just delete first angle
            if adj_angles != [0, 0, 0, 0, 0, 0] and claw_agl != 255:
                if len(angles_list) == num_to_avg:
                    del angles_list[0]
                    angles_list += [angles]
                else:
                    angles_list += [angles]
            avg_agls = np.mean(np.array(angles_list), axis=0)
            angles_dict[dir]["agl_list"] = [round(x) for x in avg_agls]

        frame1 = cv.flip(frame1, 1)
        frame0 = cv.flip(frame0, 1)

        height = 30
        for dir in ["left", "right"]:
            cv.putText(
                frame0,
                dir + "- RAW Wrist: " + str(kp_3d[dir]["raw_WRIST"]),
                (20, height),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            height += 30

        for dir in ["left", "right"]:
            for joint in joint_list:
                cv.putText(
                    frame0,
                    dir + "-" + joint + ": " + str(kp_3d[dir][joint]),
                    (20, height),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                height += 30

        height_2 = 700
        for agl in ["angles", "agl_list"]:
            for dir in ["left", "right"]:
                cv.putText(
                    frame0,
                    dir + " - %s: " % (agl) + str(angles_dict[dir][agl]),
                    (20, height_2),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                height_2 -= 30

        left_angles = angles_dict["left"]["agl_list"]
        print(left_angles)
        arduino_serial.write(
            struct.pack(
                ">BBBBBBB",
                left_angles[0],
                left_angles[1],
                left_angles[2],
                left_angles[3],
                left_angles[4],
                left_angles[5],
                left_angles[6],
            )
        )

        cv.imshow("cam1", frame1)
        cv.imshow("cam0", frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break  # 27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


if __name__ == "__main__":

    # this will load the sample videos if no camera ID is given
    # input_stream1 = 'media/cam0_test.mp4'
    # input_stream2 = 'media/cam1_test.mp4'

    # put camera id as command line arguements
    # if len(sys.argv) == 3:
    input_stream1 = 0
    input_stream2 = 1

    # get projection matrices
    param_dict = open_pickle("camera_calb/camera_parameters/cam_params.pkl")
    P0 = param_dict["cam0"]["P"]
    P1 = param_dict["cam1"]["P"]

    run_mp(input_stream1, input_stream2, P0, P1)

    # this will create keypoints file in current working folder
    # write_keypoints_to_disk("camera_calb/kp_test/kpts_cam0.dat", kpts_cam0)
    # write_keypoints_to_disk("camera_calb/kp_test/kpts_cam1.dat", kpts_cam1)
    # write_keypoints_to_disk("camera_calb/kp_test/kpts_3d.dat", kpts_3d)


#%%
