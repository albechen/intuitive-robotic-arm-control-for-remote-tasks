#%%
import numpy as np

raw_cords = {
    "top_front_left": [[0, 0, 10], [0, 0, 10], [0, 0, 10]],
    "top_front_right": [[9, 0, 10], [9, 0, 10], [9, 0, 10]],
    "top_back_left": [[0, 9, 10], [0, 9, 10], [0, 9, 10]],
    "top_back_right": [[9, 9, 10], [9, 9, 10], [9, 9, 10]],
    "bot_front_left": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "bot_front_right": [[10, 0, 0], [10, 0, 0], [10, 0, 0]],
    "bot_back_left": [[0, 10, 0], [0, 10, 0], [0, 10, 0]],
    "bot_back_right": [[10, 10, 0], [10, 10, 0], [10, 10, 0]],
}
avg_cords = raw_cords
for location in raw_cords:
    avg_cords[location] = np.mean(raw_cords[location], axis=0)
avg_cords

#%%
for z_dir in ["top", "bot"]:
    for y_dir in ["front", "back"]:
        dir_cords = avg_cords["_".join([z_dir, y_dir, "right"])]
        print(dir_cords)

#%%
front_right = avg_cords["_".join(["top", "front", "right"])]
back_right = avg_cords["_".join(["top", "back", "right"])]
front_left = avg_cords["_".join(["top", "front", "left"])]
back_left = avg_cords["_".join(["top", "back", "left"])]

front_right = [2, 0]
back_right = [0, 1]
front_left = [1, 0]
back_left = [-1, 1]

min_offset = back_left[0] - front_left[0]
max_1 = abs(front_right[0] - front_left[0])
max_2 = abs(back_right[0] - back_left[0])
max_scale = max_2 / max_1

main_start = 1
main_end = main_start * max_scale + min_offset
second_axis = 0


#%%
z_diff = []
for y_dir in ["front", "back"]:
    for x_dir in ["left", "right"]:
        top = avg_cords["top_" + y_dir + "_" + x_dir][2]
        bot = avg_cords["bot_" + y_dir + "_" + x_dir][2]
        z_diff.append(top - bot)
z_diff
#%%
print(np.mean(z_diff), np.std(z_diff))
# %%
