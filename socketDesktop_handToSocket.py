#%%
import cv2 as cv
import mediapipe as mp
import numpy as np
from time import time
from src.angle_calc.utils import DLT, get_projection_matrix, write_keypoints_to_disk
from src.angle_calc.inverse_kinematics import calculate_angles_given_joint_loc
import socket
import pickle

#%%
def get_origin_pts(P0, P1):
    # get origin cordinates
    coordinate_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points

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

    return pixel_points_camera0, pixel_points_camera1


def calculate_target_steps(steps_fullTurn: int, target_degree: float) -> int:
    steps_to_move = steps_fullTurn * round(target_degree) / 360
    return round(steps_to_move)


def run_mp(input_stream1, input_stream2, P0, P1):

    # SOCKET CONNECTION
    host = "***REMOVED***"  # client IP (desktop)
    port = 4005
    server = ("***REMOVED***", 4000)  # server IP (laptop)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    print('CONNECTED TO SERVER')

    # MEDIAPIPE CONNECTION
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    ##### DEFINED COSTANTS
    joint_list = [
        ["WRIST", 0],
        ["THUMB_TIP", 4],
        ["INDEX_FINGER_TIP", 8],
        ["PINKY_TIP", 20],
    ]

    bounds_list = [
        {"min": -90, "max": 200},
        {"min": 0, "max": 120},
        {"min": 0, "max": 135},
        {"min": -180, "max": 180},
        {"min": -120, "max": 120},
        {"min": -180, "max": 180},
        {"min": -360, "max": 360},
    ]

    lens = [8, 20, 13.5, 5.5, 0, 7]

    height = 720
    width = 1280

    ## ARRAYS TO FLIP AND ROTATE 3D POINTS
    Rx = np.array(([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]))
    Rz = np.array(([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    flip = np.array(([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))

    ## CONST FOR MOTORS
    steps_28BYJ = 2038
    steps_nema17_p = 200 * 4 * 6 * 6
    steps_nema17_c = 200 * 8 * 20

    stepper_const_list = [steps_nema17_c] + [steps_nema17_p] * 2 + [steps_28BYJ] * 4

    # input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, width)
        cap.set(4, height)

    # create hand keypoints detector object.
    hands0 = mp_hands.Hands(
        min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5
    )
    hands1 = mp_hands.Hands(
        min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5
    )

    # get origin points
    pixel_points_camera0, pixel_points_camera1 = get_origin_pts(P0, P1)

    ###### INITALIZE LISTS FOR CALCULATIONS
    kpts_3d = []
    rolling_agl_list = [[0, 0, 0, 0, 0, 0, 0]]
    num_to_avg = 4

    ####### SET INITIAL TIME
    priorTime = round(time() * 1000)

    while True:

        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = hands0.process(frame0)
        results1 = hands1.process(frame1)

        # prepare list of hand keypoints of this frame
        kpt_dict = {
            "c0": {},
            "c1": {},
            "kp_3d": {},
        }

        # frame0 kpts
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for joints in joint_list:
                    j_name = joints[0]
                    j_num = joints[1]

                    pxl_x = int(
                        round(frame0.shape[1] * hand_landmarks.landmark[j_num].x)
                    )
                    pxl_y = int(
                        round(frame0.shape[0] * hand_landmarks.landmark[j_num].y)
                    )
                    kpts = [pxl_x, pxl_y]
                    kpt_dict["c0"][j_name] = kpts

        else:
            for joints in joint_list:
                kpt_dict["c0"][joints[0]] = [-999, -999]

        # frame1 kpts
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for joints in joint_list:
                    j_name = joints[0]
                    j_num = joints[1]

                    pxl_x = int(
                        round(frame0.shape[1] * hand_landmarks.landmark[j_num].x)
                    )
                    pxl_y = int(
                        round(frame0.shape[0] * hand_landmarks.landmark[j_num].y)
                    )
                    kpts = [pxl_x, pxl_y]
                    kpt_dict["c1"][j_name] = kpts

        else:
            for joints in joint_list:
                kpt_dict["c1"][joints[0]] = [-999, -999]

        frame_p3ds = []

        for joint in joint_list:
            kp_0 = kpt_dict["c0"][joint[0]]
            kp_1 = kpt_dict["c1"][joint[0]]

            if kp_0[0] == -999 or kp_1[0] == -999:
                # kpt_dict['kp_3d']["raw_" + joint] = [-999, -999, -999]
                kpt_dict["kp_3d"][joint[0]] = [-999, -999, -999]

            else:
                kp = DLT(P0, P1, kp_0, kp_1)
                kpt_rotated = Rx @ Rz @ kp @ flip
                # kpt_dict['kp_3d']["raw_" + joint] = kp
                kpt_dict["kp_3d"][joint[0]] = kpt_rotated

        for joint in joint_list:
            frame_p3ds.append(kpt_dict["kp_3d"][joint[0]])

        """
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        """
        if any([x[0] == -999 for x in frame_p3ds]):
            pass
        else:
            frame_p3ds = np.array(frame_p3ds).reshape((4, 3))
            kpts_3d.append(frame_p3ds)

        ### GET ANGLE CACLULATIONS
        curr_agl_list = [-999, -999, -999, -999, -999, -999, -999]

        if (
            kpt_dict["kp_3d"]["WRIST"][0] == -999
            or (kpt_dict["kp_3d"]["PINKY_TIP"][0] == -999)
            or (kpt_dict["kp_3d"]["INDEX_FINGER_TIP"][0] == -999)
            or (kpt_dict["kp_3d"]["THUMB_TIP"][0] == -999)
        ):
            curr_agl_list = [-999, -999, -999, -999, -999, -999, -999]

        else:
            multp_agl_list = calculate_angles_given_joint_loc(
                kpt_dict["kp_3d"]["WRIST"],
                kpt_dict["kp_3d"]["PINKY_TIP"],
                kpt_dict["kp_3d"]["INDEX_FINGER_TIP"],
                kpt_dict["kp_3d"]["THUMB_TIP"],
                lens,
                bounds_list,
            )

            if any(any(x == -999 for x in poss_agls) for poss_agls in multp_agl_list):
                curr_agl_list = [-999, -999, -999, -999, -999, -999, -999]

            else:
                # CHECK CLOSEST ANGLES

                latest_angles = rolling_agl_list[-1]
                sum_angle_diff = [0] * 4
                for n in range(4):
                    abs_diff = [
                        abs(x - y)
                        for x, y in zip(multp_agl_list[n][3:6], latest_angles[3:6])
                    ]
                    sum_angle_diff[n] = sum(abs_diff)
                min_index = sum_angle_diff.index(min(sum_angle_diff))
                curr_agl_list = multp_agl_list[min_index]

                # AVG OUT ANGLES AND SAVE
                # catch instances where failed angle calc
                # average out certain number of angles and remove latest
                # just delete first angle
                if len(rolling_agl_list) == num_to_avg:
                    del rolling_agl_list[0]
                rolling_agl_list += [curr_agl_list]

        avg_agls = np.mean(np.array(rolling_agl_list), axis=0)
        avg_agls = [round(x, 1) for x in avg_agls]

        # Draw the hand annotations on the image.
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        ## DRAW HAND LANDMARKS
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        ## DRAW OUT ORIGIN POINTS FOR CORDINATE SYSTEM
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

        ## FLIP SO IT'S EASIER TO CONTROL
        frame1 = cv.flip(frame1, 1)
        frame0 = cv.flip(frame0, 1)

        ## READ OUT CORDINATES OF EACH HAND POINT IN 3D
        height = 30
        for joint in joint_list:
            cv.putText(
                frame0,
                "{}: {}".format(joint[0], str(kpt_dict["kp_3d"][joint[0]])),
                (20, height),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            height += 30

        height_2 = 700
        cv.putText(
            frame0,
            "Current Angles: {}".format(str(curr_agl_list)),
            (20, height_2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        height_2 -= 30

        cv.putText(
            frame0,
            "Avg Angles: {}".format(str(avg_agls)),
            (20, height_2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # delay sending data to arduino every 500 ms
        currentTime = round(time() * 1000)
        if currentTime - priorTime > 500:

            steps_list = [
                calculate_target_steps(const, angle)
                for angle, const in zip(avg_agls, stepper_const_list)
            ]

            #### SOCKET SENDING ########
            msg = pickle.dumps(steps_list)
            s.sendto(msg, server)
            print('SENT: ', msg)
            #########################

            # send data later
            priorTime = currentTime

        ## SHOW CAMERA
        cv.imshow("cam1", frame1)
        cv.imshow("cam0", frame0)

        # ESC key exits camera and resets angles to 0
        k = cv.waitKey(1)
        if k & 0xFF == 27:

            #### SOCKET SENDING ########
            steps_list = [0] * 7
            msg = pickle.dumps(steps_list)
            s.sendto(msg, server)
            print('SENT: ', msg)
            s.close()
            #########################

            break  # 27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_3d)


if __name__ == "__main__":

    # input_stream1 = 'media/cam0_test.mp4'
    # input_stream2 = 'media/cam1_test.mp4'

    input_stream1 = 0
    input_stream2 = 1

    # projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    # this will create keypoints file in current working folder
    write_keypoints_to_disk("cam_calb/data/kpts_3d_temp.dat", kpts_3d)
