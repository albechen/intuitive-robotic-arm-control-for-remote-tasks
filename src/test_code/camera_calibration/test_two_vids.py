#%%
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle


#%%

frame_shape = [1080, 1920]

###### MAIN BODY SHOWING WRITTEN ON VIDEO ###################
def run_mp(input_stream1, input_stream2):
    # input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    # for cap in caps:
    #     cap.set(3, frame_shape[1])
    #     cap.set(4, frame_shape[0])

    while 1:

        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        cv.imshow("cam1", frame1)
        cv.imshow("cam0", frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break  # 27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


if __name__ == "__main__":

    # this will load the sample videos if no camera ID is given
    # input_stream1 = 'media/cam0_test.mp4'
    # input_stream2 = 'media/cam1_test.mp4'

    # put camera id as command line arguements
    # if len(sys.argv) == 3:
    input_stream1 = 0
    input_stream2 = 1

    run_mp(input_stream1, input_stream2)

    # this will create keypoints file in current working folder
    # write_keypoints_to_disk("camera_calb/kp_test/kpts_cam0.dat", kpts_cam0)
    # write_keypoints_to_disk("camera_calb/kp_test/kpts_cam1.dat", kpts_cam1)
    # write_keypoints_to_disk("camera_calb/kp_test/kpts_3d.dat", kpts_3d)


#%%
