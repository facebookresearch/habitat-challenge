import argparse
import os

import numpy
import numpy as np
from omegaconf import DictConfig

import habitat
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default import get_config

from config import HabitatChallengeConfigPlugin


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: DictConfig):
        self._task_config = task_config

    def reset(self):
        pass

    def act(self, observations):
        if "velocity_control" in self._task_config.habitat.task.actions:
            return {
                'action': ("velocity_control", "velocity_stop"),
                'action_args': {
                    "angular_velocity": np.random.rand(1),
                    "linear_velocity": np.random.rand(1),
                    "camera_pitch_velocity": np.random.rand(1),
                    "velocity_stop": np.random.rand(1),
                }
            }
        elif "waypoint_control" in self._task_config.habitat.task.actions:
            return {
                'action': ("waypoint_control", "velocity_stop"),
                'action_args': {
                    "xyt_waypoint": np.random.rand(3),
                    "max_duration": np.random.rand(1),
                    "delta_camera_pitch_angle": np.random.rand(1),
                    "velocity_stop": np.random.rand(1),
                }
            }
        elif "move_forward_waypoint" in self._task_config.habitat.task.actions:
            return {"action": np.random.choice(self._task_config.habitat.task.actions)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="velocity_controller",
        choices=[
            "velocity_controller",
            "waypoint_controller",
            "discrete_waypoint_controller"
        ],
        help="Action space to use for the agent",
    )
    args = parser.parse_args()

    benchmark_config_path = os.environ["CHALLENGE_CONFIG_FILE"]

    register_hydra_plugin(HabitatChallengeConfigPlugin)

    config = get_config(
        benchmark_config_path,
        overrides=[
            "habitat/task/actions=" + args.action_space,
        ],
    )
    

    agent = RandomAgent(task_config=config)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False, action_space=args.action_space)
    else:
        challenge = habitat.Challenge(eval_remote=True, action_space=args.action_space)

    challenge.submit(agent)    


if __name__ == "__main__":
    main()
