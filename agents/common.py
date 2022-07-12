import gym.spaces as spaces
import numpy as np

from habitat.core.spaces import ActionSpace, EmptySpace

CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256


def get_action_space():
    return ActionSpace(
        {
            "ARM_ACTION": spaces.Dict(
                {
                    "arm_action": spaces.Box(
                        low=-1.0, high=1.0, shape=(7,), dtype=np.float32
                    ),
                    "grip_action": spaces.Box(
                        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            ),
            "BASE_VELOCITY": spaces.Dict(
                {
                    "base_vel": spaces.Box(
                        low=-20.0, high=20.0, shape=(2,), dtype=np.float32
                    )
                }
            ),
            "REARRANGE_STOP": EmptySpace(),
        }
    )
    # return spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)


def get_obs_space():
    return spaces.Dict(
        {
            "robot_head_depth": spaces.Box(
                low=0, high=1, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 1), dtype=np.float32
            ),
            "robot_head_rgb": spaces.Box(
                low=0,
                high=255,
                shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 1),
                dtype=np.float32,
            ),
            "obj_start_sensor": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,),
                dtype=np.float32,
            ),
            "obj_goal_sensor": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,),
                dtype=np.float32,
            ),
            "obj_start_gps_compass": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,),
                dtype=np.float32,
            ),
            "obj_goal_gps_compass": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,),
                dtype=np.float32,
            ),
            "joint": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(7,),
                dtype=np.float32,
            ),
            "is_holding": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(1,),
                dtype=np.float32,
            ),
            "relative_resting_position": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,),
                dtype=np.float32,
            ),
        }
    )
