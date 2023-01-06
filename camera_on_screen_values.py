#%%
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
from time import sleep, time
from src.angle_calc.utils import DLT
from src.angle_calc.inverse_kinematics import calculate_angles_given_joint_loc
from pySerialTransfer import pySerialTransfer as txfer


# link_nema17 = txfer.SerialTransfer("COM5")
# link_nema17.open()
# link_28BYJ = txfer.SerialTransfer('COM6')
# link_28BYJ.open()

sleep(3)  # allow some time for the Arduino to completely reset


def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


#%%
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
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
    {"min": -90, "max": 200},
    {"min": 0, "max": 120},
    {"min": 0, "max": 135},
    {"min": -360, "max": 360},
    {"min": -360, "max": 360},
    {"min": -360, "max": 360},
    {"min": -360, "max": 360},
]
lens = [8, 20, 13.5, 5.5, 0, 7]


def extract_xy_cords(ih, iw, cords, shouldRound=False):
    if shouldRound == False:
        return [cords.x * iw, 1, cords.y * ih]
    else:
        return [int(cords.x * iw), int(cords.y * ih)]


def get_joint_cords(results, ih, iw, joint_name):
    right_joint_cord = [-999, -999]
    left_joint_cord = [-999, -999]
    r_right_joint_cord = [-999, -999]
    r_left_joint_cord = [-999, -999]

    for handness, hand_landmarks in zip(
        results.multi_handedness, results.multi_hand_landmarks
    ):
        joint_landmark = hand_landmarks.landmark[mp_hands.HandLandmark[joint_name]]
        joint_cords = extract_xy_cords(ih, iw, joint_landmark, False)
        r_joint_cords = extract_xy_cords(ih, iw, joint_landmark, True)

        if handness.classification[0].label == "Left":
            right_joint_cord = joint_cords
            r_right_joint_cord = r_joint_cords
        elif handness.classification[0].label == "Right":
            left_joint_cord = joint_cords
            r_left_joint_cord = r_joint_cords

    return right_joint_cord, left_joint_cord, r_right_joint_cord, r_left_joint_cord


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
    angles_list_dict = {
        "left": [[0, 0, 0, 0, 0, 0, 0]],
        "right": [[0, 0, 0, 0, 0, 0, 0]],
    }
    num_to_avg = 4
    priorTime = round(time() * 1000)

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
                right, left, rr, rl = get_joint_cords(results0, height, width, joint)
                cv.circle(frame0, rr, 3, (0, 0, 255), -1)
                cv.circle(frame0, rl, 3, (0, 0, 255), -1)
                frame_dict["cam_0"]["right"][joint] = right
                frame_dict["cam_0"]["left"][joint] = left
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            for joint in joint_list:
                frame_dict["cam_0"]["right"][joint] = [-999, -999]
                frame_dict["cam_0"]["left"][joint] = [-999, -999]

        if results1.multi_hand_landmarks:
            for joint in joint_list:
                right, left, rr, rl = get_joint_cords(results1, height, width, joint)
                cv.circle(frame1, rr, 3, (0, 0, 255), -1)
                cv.circle(frame1, rl, 3, (0, 0, 255), -1)
                frame_dict["cam_1"]["right"][joint] = right
                frame_dict["cam_1"]["left"][joint] = left
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            for joint in joint_list:
                frame_dict["cam_1"]["right"][joint] = [-999, -999]
                frame_dict["cam_1"]["left"][joint] = [-999, -999]

        # calculate 3d position
        kp_3d = {"left": {}, "right": {}}
        angles_dict = {
            "left": {
                "angles": [0, 0, 0, 0, 0, 0, 0],
                "agl_list": [0, 0, 0, 0, 0, 0, 0],
            },
            "right": {
                "angles": [0, 0, 0, 0, 0, 0, 0],
                "agl_list": [0, 0, 0, 0, 0, 0, 0],
            },
        }

        for dir in ["left", "right"]:
            new_origin = [0, 0, 0]
            if dir == "left":
                new_origin = left_origin
            elif dir == "right":
                new_origin = right_origin

            for joint in joint_list:
                kp_0 = frame_dict["cam_0"][dir][joint]
                kp_1 = frame_dict["cam_1"][dir][joint]

                if kp_0 == [-999, -999] or kp_1 == [-999, -999]:
                    kp_3d[dir][joint] = [-999, -999, -999]
                    kp_3d[dir]["raw_" + joint] = [-999, -999, -999]

                else:
                    kp = DLT(P0, P1, kp_0, kp_1)
                    kp_3d[dir]["raw_" + joint] = kp
                    kp_3d[dir][joint] = offset_3d_origin(kp, new_origin)

            if (
                kp_3d[dir]["WRIST"] == [-999, -999, -999]
                or (kp_3d[dir]["PINKY_TIP"] == [-999, -999, -999])
                or (kp_3d[dir]["INDEX_FINGER_TIP"] == [-999, -999, -999])
                or (kp_3d[dir]["THUMB_TIP"] == [-999, -999, -999])
            ):
                adj_angles = [-999, -999, -999, -999, -999, -999, -999]
                angles_dict[dir]["angles"] = adj_angles

            else:
                adj_angles_list = calculate_angles_given_joint_loc(
                    kp_3d[dir]["WRIST"],
                    kp_3d[dir]["PINKY_TIP"],
                    kp_3d[dir]["INDEX_FINGER_TIP"],
                    kp_3d[dir]["THUMB_TIP"],
                    lens,
                    bounds_list,
                )

                if any(any(x == -999 for x in adj_agl) for adj_agl in adj_angles_list):
                    adj_angles = [-999, -999, -999, -999, -999, -999, -999]
                    angles_dict[dir]["angles"] = adj_angles

                else:
                    angles_list = angles_list_dict[dir]

                    # CHECK CLOSEST ANGLES
                    latest_angles = angles_list[-1]
                    sum_angle_diff = [0] * 4
                    for n in range(4):
                        abs_diff = [
                            abs(x - y)
                            for x, y in zip(adj_angles_list[n][3:6], latest_angles[3:6])
                        ]
                        sum_angle_diff[n] = sum(abs_diff)
                    min_index = sum_angle_diff.index(min(sum_angle_diff))
                    adj_angles = adj_angles_list[min_index]

                    angles_dict[dir]["angles"] = adj_angles

                    # AVG OUT ANGLES AND SAVE
                    # catch instances where failed angle calc
                    # average out certain number of angles and remove latest
                    # just delete first angle
                    if len(angles_list) == num_to_avg:
                        del angles_list[0]
                        angles_list += [adj_angles]
                    else:
                        angles_list += [adj_angles]
                    angles_list_dict[dir] = angles_list

            avg_agls = np.mean(np.array(angles_list_dict[dir]), axis=0)
            angles_dict[dir]["agl_list"] = avg_agls

        frame1 = cv.flip(frame1, 1)
        frame0 = cv.flip(frame0, 1)

        height = 30
        for dir in ["left", "right"]:
            cv.putText(
                frame0,
                dir
                + "- RAW Wrist: "
                + str([round(x, 1) for x in kp_3d[dir]["raw_WRIST"]]),
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
                    dir
                    + "-"
                    + joint
                    + ": "
                    + str([round(x, 1) for x in kp_3d[dir][joint]]),
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

        # delay sending data to arduino every 500 ms
        currentTime = round(time() * 1000)
        if currentTime - priorTime > 1000:
            steps_28BYJ = 2048
            steps_nema17_p = 200 * 4 * 6 * 6
            steps_nema17_c = 200 * 8 * 20

            def calculate_target_steps(
                steps_fullTurn: int, target_degree: float
            ) -> int:
                steps_to_move = steps_fullTurn * round(target_degree) / 360
                return round(steps_to_move)

            stepper_const_list = (
                [steps_nema17_c] + [steps_nema17_p] * 2 + [steps_28BYJ] * 4
            )

            steps_list = [
                calculate_target_steps(const, angle)
                for angle, const in zip(left_angles, stepper_const_list)
            ]

            ### https://github.com/PowerBroker2/pySerialTransfer ####
            send_size = 0

            # ###################################################################
            # # Send a list
            # ###################################################################
            # list_size = link_nema17.tx_obj(steps_list[:3])
            # send_size += list_size

            # ###################################################################
            # # Transmit all the data to send in a single packet
            # ###################################################################
            # link_nema17.send(send_size)

            # ###################################################################
            # # Wait for a response and report any errors while receiving packets
            # ###################################################################
            # while not link_nema17.available():
            #     if link_nema17.status < 0:
            #         if link_nema17.status == txfer.CRC_ERROR:
            #             print("ERROR: CRC_ERROR")
            #         elif link_nema17.status == txfer.PAYLOAD_ERROR:
            #             print("ERROR: PAYLOAD_ERROR")
            #         elif link_nema17.status == txfer.STOP_BYTE_ERROR:
            #             print("ERROR: STOP_BYTE_ERROR")
            #         else:
            #             print("ERROR: {}".format(link_nema17.status))
            # ###################################################################
            # # Parse response list
            # ###################################################################
            # rec_list_ = link_nema17.rx_obj(
            #     obj_type=type(steps_list), obj_byte_size=list_size, list_format="i"
            # )

            # ###################################################################
            # # Display the received data
            # ###################################################################
            # print(
            #     "SENT: {}    RCVD: {}".format(
            #         steps_list[:3],
            #         rec_list_,
            #     )
            # )

            # # send data later
            # priorTime = currentTime

        cv.imshow("cam1", frame1)
        cv.imshow("cam0", frame0)

        # ESC key exits camera and resets angles to 0
        k = cv.waitKey(1)
        if k & 0xFF == 27:
            send_size = 0

            # ###################################################################
            # # Send a list
            # ###################################################################
            # list_size = link_nema17.tx_obj([0, 0, 0])
            # send_size += list_size

            # ###################################################################
            # # Transmit all the data to send in a single packet
            # ###################################################################
            # link_nema17.send(send_size)

            # # ser_28BYJ.write(
            # #     struct.pack(
            # #         "4h",
            # #         0,
            # #         0,
            # #         0,
            # #         0,
            # #     )
            # # )
            break  # 27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


if __name__ == "__main__":

    # put camera id as command line arguements
    input_stream1 = 0
    input_stream2 = 1

    # get projection matrices
    param_dict = open_pickle("camera_calb/camera_parameters/cam_params.pkl")
    P0 = param_dict["cam0"]["P"]
    P1 = param_dict["cam1"]["P"]

    run_mp(input_stream1, input_stream2, P0, P1)


#%%
