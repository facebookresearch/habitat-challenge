import habitat
from habitat.sims.habitat_simulator import SimulatorActions, SIM_NAME_TO_ACTION


class ForwardOnlyAgent(habitat.Agent):
    def reset(self):
        pass

    def act(self, observations):
        action = SIM_NAME_TO_ACTION[SimulatorActions.FORWARD.value]
        return action


def main():
    agent = ForwardOnlyAgent()
    challenge = habitat.Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()
