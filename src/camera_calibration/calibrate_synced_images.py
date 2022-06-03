#%%
import cv2
import threading
from datetime import datetime, timedelta
import glob

#%%
class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)


def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    width = 1920
    height = 1080
    cam.set(3, width)  # 1280 x 720
    cam.set(4, height)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)

        if key == ord("i"):
            now = datetime.now()
            current_time = now.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                "calibration_images/c%s_idv/%s.png" % (camID, current_time), frame
            )

        if key == ord("s"):
            now = datetime.now()
            current_time = now.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                "calibration_images/c%s_syc/%s.png" % (camID, current_time), frame
            )

        if key == 32:
            now = datetime.now()
            current_time = now.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                "calibration_images/c%s_hand/%s.png" % (camID, current_time), frame
            )

        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)


# Create two threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread1.start()
thread2.start()

# %%
def check_synced_images():
    images_0 = sorted(glob.glob("calibration_images/c0_syc/*"))
    images_1 = sorted(glob.glob("calibration_images/c1_syc/*"))
    print("Cam 0 Count: ", len(images_0))
    print("Cam 1 Count: ", len(images_1))

    mm_c0 = []
    mm_c1 = []
    for f0, f1 in zip(images_0, images_1):
        sec_1 = datetime(100, 1, 1, 1, int(f1[37:-6]), int(f1[39:-4]))
        sec_0 = datetime(100, 1, 1, 1, int(f0[37:-6]), int(f0[39:-4]))
        s0_add1 = sec_0 + timedelta(seconds=1)
        s0_rmv1 = sec_0 - timedelta(seconds=1)

        if sec_1 >= s0_rmv1 and sec_1 <= s0_add1:
            pass
        else:
            print(sec_1, s0_add1, s0_rmv1)
            mm_c0.append(f0)
            mm_c1.append(f1)
    print(mm_c0)
    print(mm_c1)


check_synced_images()
#%%


import cv2


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

    if k == 27:
        break
    elif k == ord("s"):  # wait for 's' key to save and exit
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite("calibration_images/c%s_syc/%s.png" % (0, dt), frame0)
        cv2.imwrite("calibration_images/c%s_syc/%s.png" % (1, dt), frame1)
        print("images saved!")

    cv2.imshow("cam0", frame0)
    cv2.imshow("cam1", frame1)
