import cv2
import numpy as np

frame_shape = [1080, 1920]
cap0 = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

cap.set(3, frame_shape[1])
cap.set(4, frame_shape[0])

cap0.set(3, frame_shape[1])
cap0.set(4, frame_shape[0])

while 1:
    ret0, frame0 = cap0.read()
    ret, frame = cap.read()
    # print(height)
    
    # cv2.imshow("Cropped Image", crop_img)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", frame)
    cv2.imshow("frame0", frame0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows