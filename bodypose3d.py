import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from src.camera_calibration.utils import (
    DLT,
    get_projection_matrix,
    write_keypoints_to_disk,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

frame_shape = [1080, 1920]

# add here if you need more keypoints

joint_list = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
]


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

        if handness.classification[0].label == "Right":
            right_joint_cord = joint_cords
        elif handness.classification[0].label == "Left":
            left_joint_cord = joint_cords

    return right_joint_cord, left_joint_cord


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

    # containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    while True:

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

        # check for keypoints detection
        frame0_left = []
        frame0_right = []
        if results0.multi_hand_landmarks:
            for joint in joint_list:
                right, left = get_joint_cords(results0, height, width, joint)
                cv.circle(frame0, right, 3, (0, 0, 255), -1)
                cv.circle(frame0, left, 3, (0, 0, 255), -1)
                frame0_right.append(right)
                frame0_left.append(left)
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_right = [[-1, -1]] * len(joint_list)
            frame0_left = [[-1, -1]] * len(joint_list)

        frame0_both = frame0_right + frame0_left

        # this will keep keypoints of this frame in memory
        kpts_cam0.append(frame0_both)

        frame1_left = []
        frame1_right = []
        if results1.multi_hand_landmarks:
            for joint in joint_list:
                right, left = get_joint_cords(results1, height, width, joint)
                cv.circle(frame1, right, 3, (0, 0, 255), -1)
                cv.circle(frame1, left, 3, (0, 0, 255), -1)
                frame1_right.append(right)
                frame1_left.append(left)
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_right = [[-1, -1]] * len(joint_list)
            frame1_left = [[-1, -1]] * len(joint_list)

        frame1_both = frame1_right + frame1_left
        # update keypoints container
        kpts_cam1.append(frame1_both)

        # calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_both, frame1_both):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)  # calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        """
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        """
        frame_p3ds = np.array(frame_p3ds)  # .reshape((12, 3))
        kpts_3d.append(frame_p3ds)

        cv.putText(
            frame0,
            str(frame_p3ds[0]),
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv.imshow("cam1", frame1)
        cv.imshow("cam0", frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break  # 27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


if __name__ == "__main__":

    # this will load the sample videos if no camera ID is given
    # input_stream1 = 'media/cam0_test.mp4'
    # input_stream2 = 'media/cam1_test.mp4'

    # put camera id as command line arguements
    # if len(sys.argv) == 3:
    input_stream1 = 0
    input_stream2 = 1

    # get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    # this will create keypoints file in current working folder
    write_keypoints_to_disk("camera_calb/kp_test/kpts_cam0.dat", kpts_cam0)
    write_keypoints_to_disk("camera_calb/kp_test/kpts_cam1.dat", kpts_cam1)
    write_keypoints_to_disk("camera_calb/kp_test/kpts_3d.dat", kpts_3d)
