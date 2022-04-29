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
def extract_x_y_z_cords(cords):
    return np.array([cords.x, cords.y, cords.z])


def get_wrist_cords(results):
    right_wrist_cord = np.array([0, 0, 0])
    left_wrist_cord = np.array([0, 0, 0])

    for handness, hand_landmarks in zip(
        results.multi_handedness, results.multi_hand_world_landmarks
    ):
        wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark["WRIST"]]
        wrist_cords = extract_x_y_z_cords(wrist_landmark)

        if handness.classification[0].label == "Right":
            right_wrist_cord = wrist_cords
        elif handness.classification[0].label == "Left":
            left_wrist_cord = wrist_cords

    return right_wrist_cord, left_wrist_cord


def rolling_average_wrist_cord(new_cord, list_cords, max_samples):
    list_cords = np.concatenate((list_cords, new_cord[None, :]), axis=0)
    average_cords = np.mean(list_cords, axis=0)
    if len(list_cords) == max_samples:
        list_cords = np.delete(list_cords, 0)
    return average_cords, list_cords


# START VIDEO
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

pastTime = time.time() * 1000
rw_cord_list = np.array([[0, 0, 0]])
lw_cord_list = np.array([[0, 0, 0]])

with mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)
        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            rw_cord, lw_cord = get_wrist_cords(results)
        else:
            rw_cord = np.array([0, 0, 0])
            lw_cord = np.array([0, 0, 0])
        arw_cord, rw_cord_list = rolling_average_wrist_cord(rw_cord, rw_cord_list, 30)
        alw_cord, rw_cord_list = rolling_average_wrist_cord(lw_cord, lw_cord_list, 30)

        wrist_str = str("WRIST " + " ".join([str(x) for x in arw_cord]))
        currentTime = time.time() * 1000
        if currentTime - pastTime > 250:
            pastTime = currentTime
            print(arw_cord)
        # rightIndex = degreeArray[6]
        # if rightIndex == [0, 0, 0]:
        #     pass
        # else:
        #     arduinoData.write(
        #         struct.pack(">BBB", rightIndex[1], rightIndex[3], rightIndex[5])
        #     )
        # print(arduinoData.readline())

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        cv2.flip(image, 1)
        cv2.putText(
            image,
            wrist_str,
            (20, 30),
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

#%%