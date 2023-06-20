# %%
import moviepy.editor as mp


# %%
def convert_mp4_to_gif(vid_name, fps, fuzz, output_width):
    mp4_path = "images/original/{}.mp4".format(vid_name)
    gif_path = "images/{}.gif".format(vid_name)

    video = mp.VideoFileClip(mp4_path)
    width = video.size[0]
    height = video.size[1]
    new_ratio = output_width / width
    new_height = new_ratio * height
    video = video.resize(height=new_height, width=output_width)  # type: ignore
    video.write_gif(gif_path, fps=fps, fuzz=fuzz)


fps = 10
fuzz = 10
output_width = 1500

# %%
vid_name = "integration_remote"
convert_mp4_to_gif(vid_name, fps, fuzz, output_width)

# %%
vid_name = "hand_detection_3d"
convert_mp4_to_gif(vid_name, fps, fuzz, output_width)

# %%
vid_name = "arm_design_version"
convert_mp4_to_gif(vid_name, fps, fuzz, output_width)
# %%
vid_name = "final_red_green"
convert_mp4_to_gif(vid_name, fps, fuzz, output_width)
# %%
vid_name = "final_bread"
convert_mp4_to_gif(vid_name, fps, fuzz, output_width)
