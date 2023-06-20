from moviepy.editor import *


def convert_mp4_to_gif(mp4_file, gif_file):
    video = VideoFileClip(mp4_file)
    video.write_gif(gif_file)


# Usage example
convert_mp4_to_gif("input.mp4", "output.gif")
