#%%
from numpy import array

test_dict = {
    "Right": {
        "c0": {
            "WRIST": [790, 343],
            "INDEX_FINGER_MCP": [740, 234],
            "MIDDLE_FINGER_MCP": [766, 225],
            "MIDDLE_FINGER_TIP": [747, 111],
            "PINKY_MCP": [815, 233],
        },
        "c1": {
            "WRIST": [851, 397],
            "INDEX_FINGER_MCP": [810, 302],
            "MIDDLE_FINGER_MCP": [829, 293],
            "MIDDLE_FINGER_TIP": [800, 197],
            "PINKY_MCP": [865, 295],
        },
        "kp_3d": {
            "WRIST": array([-13.32613232, -16.27925589, 29.2025275]),
            "INDEX_FINGER_MCP": array([-8.58129079, -18.46525851, 42.69692911]),
            "MIDDLE_FINGER_MCP": array([-11.54594865, -16.79742085, 42.45143665]),
        },
    },
    "Left": {
        "c0": {
            "WRIST": [790, 343],
            "INDEX_FINGER_MCP": [740, 234],
            "MIDDLE_FINGER_MCP": [766, 225],
            "MIDDLE_FINGER_TIP": [747, 111],
            "PINKY_MCP": [815, 233],
        },
        "c1": {
            "WRIST": [851, 397],
            "INDEX_FINGER_MCP": [810, 302],
            "MIDDLE_FINGER_MCP": [829, 293],
            "MIDDLE_FINGER_TIP": [800, 197],
            "PINKY_MCP": [865, 295],
        },
        "kp_3d": {
            "WRIST": array([-13.32613232, -16.27925589, 29.2025275]),
            "INDEX_FINGER_MCP": array([-8.58129079, -18.46525851, 42.69692911]),
            "MIDDLE_FINGER_MCP": array([-11.54594865, -16.79742085, 42.45143665]),
            "test": array([-999, -999, -999]),
        },
    },
}

any(
    any(
        any(kpt == -999 for kpt in test_dict[dir]["kp_3d"][joint])
        for joint in test_dict[dir]["kp_3d"]
    )
    for dir in test_dict
)
