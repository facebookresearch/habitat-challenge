import argparse
import habitat
import random
import numpy
import os

class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS

    def reset(self):
        pass

    def act(self, observations):
        return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}

def main():
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = RandomAgent(task_config=config)
    challenge = habitat.Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()

