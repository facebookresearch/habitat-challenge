import argparse
import habitat
import random

NUM_ACTIONS = 5

class RandomAgent(habitat.Agent):
    def reset(self):
        pass

    def act(self, observations):
        return random.randint(0, NUM_ACTIONS)

def main():
    agent = RandomAgent()
    challenge = habitat.Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()

