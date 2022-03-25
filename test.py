#%%
import cv2
import mediapipe as mp
import numpy as np
import serial

#%%
def get_cords_from_joint_name(results, fingerJoint, image_height, image_width):
    cords = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark[fingerJoint]]
    cordList = np.array([cords.x, cords.y, cords.z])
    return cordList


def calculate_angle(threeCordsArray):
    a, b, c = threeCordsArray[0], threeCordsArray[1], threeCordsArray[2]
    ab = a - b
    cb = c - b
    mag_ab = np.sqrt(ab.dot(ab))
    mag_cb = np.sqrt(cb.dot(cb))
    dotP = ab.dot(cb)
    degree = np.degrees(np.arccos(dotP / (mag_ab * mag_cb)))
    return round(degree)


def calc_angles_for_joints(results, fingerList, jointList, image_height, image_width):
    if results.left_hand_landmarks == None:
        return []

    wristCord = get_cords_from_joint_name(results, "WRIST", image_height, image_width)
    degreeArray = []

    for finger in fingerList:
        cordArray = [wristCord]
        degreeList = []
        for joint in jointList:
            fingerJoint = "_".join([finger, joint])
            cordList = get_cords_from_joint_name(
                results, fingerJoint, image_height, image_width
            )
            cordArray.append(cordList)
        for i in range(len(cordArray) - 2):
            degree = calculate_angle((cordArray[i : 3 + i]))
            degreeList.append(degree)
        degreeArray.append(degreeList)

    return degreeArray


fingerList = ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
jointList = ["MCP", "PIP", "DIP", "TIP"]
degreeArray = calc_angles_for_joints(results, fingerList, jointList, 0, 0)

tb_fingerList = ["THUMB"]
tb_jointList = ["CMC", "MCP", "IP", "TIP"]
tb_degreeArray = calc_angles_for_joints(results, tb_fingerList, tb_jointList, 0, 0)

for x, y in zip(tb_fingerList, tb_degreeArray):
    print(x, y)
for x, y in zip(fingerList, degreeArray):
    print(x, y)

#%%
# START VIDEO
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
mp_drawing.draw_landmarks

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2
) as holistic:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image_height, image_width, _ = image.shape
        # Make Detections
        results = holistic.process(image)

        # calculate values
        fingerList = ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
        jointList = ["MCP", "PIP", "DIP", "TIP"]
        degreeArray = calc_angles_for_joints(
            results, fingerList, jointList, image_height, image_width
        )

        # pose_landmarks, left_hand_landmarks, right_hand_landmarks
        # print(results.left_hand_landmarks)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Right hand
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

        # 3. Left Hand
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        # flip image
        cv2.flip(image, 1)

        # add angles
        pos = 0
        for x, y in zip(fingerList, degreeArray):
            y = " ".join([str(elem) for elem in y])
            pos += 30
            cv2.putText(
                image,
                x + " " + y,
                (20, pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Raw Webcam Feed", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

# %%
