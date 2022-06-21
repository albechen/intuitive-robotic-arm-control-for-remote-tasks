#%%
import cv2
import mediapipe as mp
import glob
import pickle

#%%
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def extract_x_y_z_cords(ih, iw, cords):
    return [cords.x * iw, cords.y * ih]


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

    return {"right": right_joint_cord, "left": left_joint_cord}


def gather_static_points_on_hand(path, joint_list):
    IMAGE_FILES = sorted(glob.glob(path + "/*"))
    hand_points = {}
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            # print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            ih, iw, _ = image.shape
            if results.multi_hand_landmarks:
                points_list = {}
                for joint in joint_list:
                    points_list[joint] = get_joint_cords(results, ih, iw, joint)
                    hand_points["pic_" + str(idx)] = points_list

            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                #   print('hand_landmarks:', hand_landmarks)

                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            file_name = r"{}".format(file).split("\\")[-1]
            cv2.imwrite(path + "_annotated/" + file_name, annotated_image)
            # Draw hand world landmarks.
            # if not results.multi_hand_world_landmarks:
            #     continue
            # for hand_world_landmarks in results.multi_hand_world_landmarks:
            #     mp_drawing.plot_landmarks(
            #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5
            #     )
    return hand_points


def transpose_hand_joint_positiion(cam_dict):
    reorg_dict = {}
    for cam in cam_dict:
        reorg_dict[cam] = {}
        for pic in cam_dict[cam]:
            reorg_dict[cam][pic] = {}
            left_list = []
            right_list = []

            for joint in cam_dict[cam][pic]:
                right_list.append(cam_dict[cam][pic][joint]["right"])
                left_list.append(cam_dict[cam][pic][joint]["left"])

            reorg_dict[cam][pic]["left"] = left_list
            reorg_dict[cam][pic]["right"] = right_list

    return reorg_dict


def gather_cam_0_1_points(base_path, cam0_str, cam1_str, joint_list):
    cam_dict = {}
    cam_dict["c0"] = gather_static_points_on_hand(base_path + cam0_str, joint_list)
    cam_dict["c1"] = gather_static_points_on_hand(base_path + cam1_str, joint_list)
    cam_dict = transpose_hand_joint_positiion(cam_dict)

    with open("calibration_matrix/annotated_points.pkl", "wb") as handle:
        pickle.dump(cam_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cam_dict


# %%
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

cam_dict = gather_cam_0_1_points(
    base_path="images/",
    cam0_str="c0_hand",
    cam1_str="c1_hand",
    joint_list=joint_list,
)
#%%
import pickle

with open("calibration_matrix/annotated_points.pkl", "rb") as handle:
        all_points = pickle.load(handle)