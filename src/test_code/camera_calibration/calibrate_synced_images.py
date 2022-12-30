#%%
import cv2
from datetime import datetime

cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)
width = 1920
height = 1080
cam0.set(3, width)  # 1280 x 720
cam0.set(4, height)
cam1.set(3, width)  # 1280 x 720
cam1.set(4, height)

while cam0.isOpened():

    rval0, frame0 = cam0.read()
    rval1, frame1 = cam1.read()

    k = cv2.waitKey(5)

    if k == 27:  # esc
        break
    elif k == ord("s"):  # s
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite("images/c%s_syc/%s.png" % (0, dt), frame0)
        cv2.imwrite("images/c%s_syc/%s.png" % (1, dt), frame1)
    elif k == 32:  # space
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite("images/c%s_hand/%s.png" % (0, dt), frame0)
        cv2.imwrite("images/c%s_hand/%s.png" % (1, dt), frame1)

    cv2.imshow("cam0", cv2.flip(frame0, 1))
    cv2.imshow("cam1", cv2.flip(frame1, 1))

cv2.destroyAllWindows()

# %%
