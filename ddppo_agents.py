#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import random
from typing import Dict, Optional

import numba
import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

import habitat
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs


from clip_policy import PointNavResNetPolicy

@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class PPOAgent(Agent):
    def __init__(self, config: Config) -> None:
        image_size = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER
        if "ObjectNav" in config.TASK_CONFIG.TASK.TYPE:
            OBJECT_CATEGORIES_NUM = 5 # 5+1 will be used internally
            spaces = {
                "objectgoal": Box(
                    low=0, high=OBJECT_CATEGORIES_NUM, shape=(1,), dtype=np.int64
                ),
                "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                "gps": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        else:
            spaces = {
                "pointgoal": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                )
            }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(image_size.HEIGHT, image_size.WIDTH, 1),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(image_size.HEIGHT, image_size.WIDTH, 3),
                dtype=np.uint8,
            )
        observation_spaces = SpaceDict(spaces)
        action_spaces = (
            Discrete(6) if "ObjectNav" in config.TASK_CONFIG.TASK.TYPE else Discrete(4)
        )
        self.obs_transforms = get_active_obs_transforms(config)
        observation_spaces = apply_obs_transforms_obs_space(
            observation_spaces, self.obs_transforms
        )

        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.RL.PPO.hidden_size

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        _seed_numba(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        policy = baseline_registry.get_policy(config.RL.POLICY.name)
        
        # self.actor_critic = policy.from_config(
        #     config, observation_spaces, action_spaces
        # )

        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_spaces.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
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
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def act(self, observations: Observations) -> Dict[str, int]:
        batch = batch_obs([observations], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        with torch.no_grad():
            (_, actions, _, self.test_recurrent_hidden_states) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore

        return {"action": actions[0][0].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type", default="blind", choices=["blind", "rgb", "depth", "rgbd"]
    )
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()

    config = get_config(
        "configs/ddppo_objectnav.yaml", ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()
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
