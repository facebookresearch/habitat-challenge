import argparse
from habitat_baselines.agents.simple_agents import get_agent_cls
import habitat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-class", type=str, default="GoalFollower")
    args = parser.parse_args()

    agent = get_agent_cls(args.agent_class)(success_distance=0.2)
    challenge = habitat.Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()

