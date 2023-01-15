#%%
import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

#%%

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

height = 720
width = 1280
# frame_shape = [720, 1280]

joint_list = [
    ["WRIST", 0],
    ["THUMB_TIP", 4],
    ["INDEX_FINGER_TIP", 8],
    ["PINKY_TIP", 20],
]


def run_mp(input_stream1, input_stream2, P0, P1):
    # input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, width)
        cap.set(4, height)

    vid0 = cv.VideoWriter(
        "media/cam0_temp.mp4",
        cv.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )
    vid1 = cv.VideoWriter(
        "media/cam1_temp.mp4",
        cv.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )

    # create hand keypoints detector object.
    hands0 = mp_hands.Hands(
        min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5
    )
    hands1 = mp_hands.Hands(
        min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5
    )

    # containers for detected keypoints for each camera
    kpts_3d = []
    kpts_cam0 = []
    kpts_cam1 = []
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

        # calculate 3d position
        # frame_p3ds = []
        # for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
        #     if uv1[0] == -1 or uv2[0] == -1:
        #         break
        #     else:
        #         _p3d = DLT(P0, P1, uv1, uv2)  # calculate 3d position of keypoint
        #     frame_p3ds.append(_p3d)
        frame_p3ds = []
        frame_kp1 = []
        frame_kp0 = []

        for joint in joint_list:
            kp_0 = kpt_dict["c0"][joint[0]]
            kp_1 = kpt_dict["c1"][joint[0]]

            if kp_0[0] == -999 or kp_1[0] == -999:
                # kpt_dict['kp_3d']["raw_" + joint] = [-999, -999, -999]
                kpt_dict["kp_3d"][joint[0]] = [-999, -999, -999]

            else:
                kp = DLT(P0, P1, kp_0, kp_1)
                # kpt_dict['kp_3d']["raw_" + joint] = kp
                kpt_dict["kp_3d"][joint[0]] = kp

        for joint in joint_list:
            frame_p3ds.append(kpt_dict["kp_3d"][joint[0]])
            frame_kp0.append(kpt_dict["c0"][joint[0]])
            frame_kp1.append(kpt_dict["c1"][joint[0]])

        """
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        """
        if any([x[0] == -999 for x in frame_p3ds]):
            pass
        else:
            frame_p3ds = np.array(frame_p3ds).reshape((4, 3))
            kpts_3d.append(frame_p3ds)

            frame_kp0 = np.array(frame_kp0).reshape((4, 2))
            kpts_cam0.append(frame_kp0)
            frame_kp1 = np.array(frame_kp1).reshape((4, 2))
            kpts_cam1.append(frame_kp1)

        # Draw the hand annotations on the image.
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

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

        vid1.write(frame1)
        vid0.write(frame0)
        cv.imshow("cam1", frame1)
        cv.imshow("cam0", frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break  # 27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()
    vid1.release()
    vid0.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


if __name__ == "__main__":

    # input_stream1 = 'media/cam0_test.mp4'
    # input_stream2 = 'media/cam1_test.mp4'

    input_stream1 = 0
    input_stream2 = 1

    # projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    # this will create keypoints file in current working folder
    write_keypoints_to_disk("data/kpts_cam0_temp.dat", kpts_cam0)
    write_keypoints_to_disk("data/kpts_cam1_temp.dat", kpts_cam1)
    write_keypoints_to_disk("data/kpts_3d_temp.dat", kpts_3d)
