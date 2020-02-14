import argparse
import habitat

NUM_ACTIONS = 6

class RandomAgent(habitat.Agent):
    def act(self, observations):
        return random.randint(NUM_ACTIONS)

def main():
    agent = RandomAgent
    challenge = habitat.Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()

