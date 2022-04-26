#%%
import cv2
import mediapipe as mp
import numpy as np

import serial
import time

from time import sleep
import struct

# arduinoData = serial.Serial("COM5", 9600)
#%%


def get_cords_from_joint_name(resultLandmark, fingerJoint, image_height, image_width):
    cords = resultLandmark.landmark[mp_holistic.HandLandmark[fingerJoint]]
    cordList = np.array(
        [cords.x * image_width, cords.y * image_height, cords.z * image_width]
    )
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


def map_degree_to_serial_output(degree):
    minDetectDegree = 60
    maxDetectDegree = 180
    if degree > maxDetectDegree:
        degree = maxDetectDegree
    elif degree < minDetectDegree:
        degree = minDetectDegree
    ratio = (degree - minDetectDegree) / (maxDetectDegree - minDetectDegree)
    inverseRatio = (1 - ratio) * 100
    return round(inverseRatio)


def calc_angles_for_joints(
    results, fingerList, jointList, tb_jointList, image_height, image_width,
):
    degreeArray = []
    fingerArray = ["LEFT_" + finger for finger in fingerList] + [
        "RIGHT_" + finger for finger in fingerList
    ]

    for resultLandmark in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if resultLandmark == None:
            for n in fingerList:
                degreeArray.append([0, 0, 0])
        else:
            wristCord = get_cords_from_joint_name(
                resultLandmark, "WRIST", image_height, image_width
            )

            for finger in fingerList:
                if finger == "THUMB":
                    tmp_jointList = tb_jointList
                else:
                    tmp_jointList = jointList
                cordArray = [wristCord]
                degreeList = []
                for joint in tmp_jointList:
                    fingerJoint = "_".join([finger, joint])
                    cordList = get_cords_from_joint_name(
                        resultLandmark, fingerJoint, image_height, image_width
                    )
                    cordArray.append(cordList)
                for i in range(len(cordArray) - 2):
                    degree = calculate_angle((cordArray[i : 3 + i]))
                    mappedDegree = map_degree_to_serial_output(degree)
                    degreeList.append(degree)
                    degreeList.append(mappedDegree)
                degreeArray.append(degreeList)

    return degreeArray, fingerArray


# fingerList = ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
# jointList = ["MCP", "PIP", "DIP", "TIP"]
# degreeArray = calc_angles_for_joints(results, fingerList, jointList, 0, 0)

# tb_fingerList = ["THUMB"]
# tb_jointList = ["CMC", "MCP", "IP", "TIP"]
# tb_degreeArray = calc_angles_for_joints(results, tb_fingerList, tb_jointList, 0, 0)

# for x, y in zip(tb_fingerList, tb_degreeArray):
#     print(x, y)
#

#%%
# START VIDEO
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
mp_drawing.draw_landmarks

cap = cv2.VideoCapture(0)

pastTime = time.time() * 1000
priorAngles = [0, 0, 0]

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
        wrist_cords = results.left_hand_landmarks.landmark[
            mp_holistic.HandLandmark["WRIST"]
        ]
        wrist_xyz = np.array([wrist_cords.x, wrist_cords.y, wrist_cords.z])
        currentTime = time.time() * 1000
        if currentTime - pastTime > 250:
            pastTime = currentTime
            print(wrist_xyz)
            # rightIndex = degreeArray[6]
            # if rightIndex == [0, 0, 0]:
            #     pass
            # else:
            #     arduinoData.write(
            #         struct.pack(">BBB", rightIndex[1], rightIndex[3], rightIndex[5])
            #     )
            # print(arduinoData.readline())

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Right hand
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
        )

        # 3. Left Hand
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
        )

        # flip image
        cv2.flip(image, 1)

        # wirst posiiton
        
        cv2.putText(
            image,
            wrist_cords,
            20,
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

#%%q
