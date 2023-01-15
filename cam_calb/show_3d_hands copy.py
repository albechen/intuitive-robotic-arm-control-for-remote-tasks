import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
from PIL import Image
import contextlib
import glob
import os


def read_keypoints(filename):
    fin = open(filename, "r")

    kpts = []
    while True:
        line = fin.readline()
        if line == "":
            break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (4, -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):

    # Rz = np.array(([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    # Rx = np.array(([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]))
    Rx = np.array(([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]))
    Rz = np.array(([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    flip = np.array(([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))

    p3ds_rotated = []
    for frame in p3ds:
        frame_kpts_rotated = []
        for kpt in frame:
            # kpt_rotated = Rx @ Rz @ kpt @ flip
            kpt_rotated = kpt
            frame_kpts_rotated.append(kpt_rotated)
        p3ds_rotated.append(frame_kpts_rotated)

    """this contains 3d points of each frame"""
    p3ds_rotated = np.array(p3ds_rotated)

    """Now visualize in 3D"""
    thumb_f = [[0, 1]]
    index_f = [[0, 2]]
    pinkie_f = [[0, 3]]
    fingers = [pinkie_f, index_f, thumb_f]
    fingers_colors = ["red", "blue", "green"]

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, kpts3d in enumerate(p3ds_rotated):
        # if i % 2 == 0:
        #     continue  # skip every 2nd frame
        for finger, finger_color in zip(fingers, fingers_colors):
            for _c in finger:
                ax.plot(
                    xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]],
                    ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
                    zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]],
                    linewidth=4,
                    c=finger_color,
                )

        # draw axes
        ax.plot(xs=[0, 5], ys=[0, 0], zs=[0, 0], linewidth=2, color="red")
        ax.plot(xs=[0, 0], ys=[0, 5], zs=[0, 0], linewidth=2, color="blue")
        ax.plot(xs=[0, 0], ys=[0, 0], zs=[0, 5], linewidth=2, color="black")

        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-10, 50)
        ax.set_xlabel("x")
        ax.set_ylim3d(-10, 50)
        ax.set_ylabel("y")
        ax.set_zlim3d(-10, 50)
        ax.set_zlabel("z")
        # ax.elev = 0.2 * i
        # ax.azim = 0.2 * i
        plt.savefig("media/pics/fig_" + str(i) + ".png")
        plt.pause(0.01)
        ax.cla()


def create_gif():

    # filepaths
    fp_in = "media/pics/fig_*.png"
    fp_out = "media/fig_temp.gif"

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(fp_in)))

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=300,
            loop=0,
        )

    delete_dir = "media/pics"
    for f in os.listdir(delete_dir):
        os.remove(os.path.join(delete_dir, f))


if __name__ == "__main__":

    p3ds = read_keypoints("data/kpts_3d_temp.dat")
    visualize_3d(p3ds)
    create_gif()
