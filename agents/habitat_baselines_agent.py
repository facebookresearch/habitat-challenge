#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import random
from typing import Dict, Optional

import gym.spaces as spaces
import numba
import numpy as np
import torch

import habitat
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.core.spaces import ActionSpace, EmptySpace
from habitat.utils.gym_adapter import continuous_vector_action_to_hab_dict
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs, get_num_actions

random_generator = np.random.RandomState()

CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    _seed_numba(seed)
    torch.random.manual_seed(seed)


def sample_random_seed():
    set_random_seed(random_generator.randint(2**32))


class PPOAgent(Agent):
    def __init__(self, config: Config) -> None:
        obs_space = spaces.Dict(
            {
                "robot_head_depth": spaces.Box(
                    low=0,
                    high=1,
                    shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 1),
                    dtype=np.float32,
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
                    shape=(2,),
                    dtype=np.float32,
                ),
                "obj_goal_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
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
        self.action_space = ActionSpace(
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

        if config.INPUT_TYPE == "blind":
            del obs_space.spaces["robot_head_depth"]
            del obs_space.spaces["robot_head_rgb"]
        elif config.INPUT_TYPE == "depth":
            del obs_space.spaces["robot_head_rgb"]
        elif config.INPUT_TYPE == "rgb":
            del obs_space.spaces["robot_head_depth"]

        self.obs_transforms = get_active_obs_transforms(config)
        obs_space = apply_obs_transforms_obs_space(obs_space, self.obs_transforms)

        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.RL.PPO.hidden_size
        random_generator.seed(config.RANDOM_SEED)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        policy = baseline_registry.get_policy(config.RL.POLICY.name)
        self.actor_critic = policy.from_config(
            config,
            obs_space,
            spaces.Box(
                shape=(get_num_actions(self.action_space),),
                low=-1,
                high=1,
                dtype=np.float32,
            ),
            orig_action_space=self.action_space,
        )

        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating a random model."
            )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(
            1,
            get_num_actions(self.action_space),
            dtype=torch.float32,
            device=self.device,
        )

    def act(self, observations: Observations) -> Dict[str, int]:
        sample_random_seed()
        batch = batch_obs([observations], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        with torch.no_grad():
            (_, actions, _, self.test_recurrent_hidden_states,) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore

        step_action = continuous_vector_action_to_hab_dict(
            self.action_space, None, actions[0].cpu().numpy()
        )

        return step_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument("--cfg-path", type=str, required=True)

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()

    config = get_config(args.cfg_path, ["BASE_TASK_CONFIG_PATH", config_paths]).clone()
    config.defrost()
    config.PTH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path

    config.RANDOM_SEED = 7
    config.freeze()

    agent = PPOAgent(config)
    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
        challenge._env.seed(config.RANDOM_SEED)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
