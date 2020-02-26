<p align="center">
  <img width = "50%" src='res/img/habitat_logo_with_text_horizontal_blue.png' />
  </p>

--------------------------------------------------------------------------------

# Habitat-Challenge

This repository contains starter code for the 2020 challenge, details of the tasks, and training and evaluation setups. For an overview of habitat-challenge visit aihabitat.org/challenge.

This year, we are hosting challenges on two embodied navigation tasks: 
PointNav (‘Go 5m north, 3m west relative to start’)
ObjectNav (‘find a chair’). 

Task #1: PointNav focuses on realism and sim2real predictivity (the ability to predict the performance of a nav-model on a real robot from its performance in simulation). 

Task #2: ObjectNav focuses on egocentric object/scene recognition and a commonsense understanding of object semantics (where is a fireplace typically located in a house?). 


## Task 1: PointNav 
In PointNav, an agent is spawned at a random starting position and orientation in an unseen environment and and asked to navigate to target coordinates specified relative to the agent’s start location (‘Go 5m north, 3m west relative to start’). No ground-truth map is available and the agent must only use its sensory input (an RGB-D camera) to navigate. 

We use Gibson 3D scenes [2] for the challenge. We use the splits provided by the Gibson dataset, retaining the train, and val sets, and separating the test set into test-standard and test-challenge. The train and val scenes are provided to participants. The test scenes are used for the official challenge evaluation and are not provided to participants.

After calling the STOP action, the agent is evaluated using the "Success weighted by Path Length" (SPL) metric [3]. An episode is deemed successful if on calling the STOP action, the agent is within 0.36m of the goal position. The evaluation will be carried out in completely new houses which are not present in training and validation splits.

### New in 2020
The main emphasis in 2020 is on increased realism and on sim2real predictivity (the ability to predict performance on a real robot from its performance in simulation). 

Specifically, we introduce the following changes inspired by our experiments and findings in \cite{kadian_sim2real}: 

1. **No GPS+Compass sensor**: In 2019, the relative coordinates specifying the goal were continuously updated during agent movement -- essentially simulating an agent with perfect localization and heading estimation (e.g. an agent with an idealized GPS+Compass). However, high-precision localization in indoor environments can not be assumed in realistic settings -- GPS has low precision indoors, (visual) odometry may be noisy, SLAM-based localization can fail, etc. Hence, in 2020's challenge the agent does NOT have a GPS+Compass sensor and must navigate solely using an egocentric RGB-D camera. This change elevates the need to perform RGBD-based online localization. 

1. **Noisy Actuation and Sensing**: In 2019, the agent actions were deterministic -- i.e. when the agent executes **turn-left 30 degrees**, it turns **exactly** 30 degrees, and forward 0.25 m moves the agent **exactly** 0.25 m forward (modulo collisions). However, no robot moves deterministically -- actuation error, surface properties such as friction, and a myriad of other sources of error introduce significant drift over a long trajectory. To model this, we introduce a noise model acquired by benchmarking a real robotic platform \cite{pyrobot2019}. \figref{fig:noisy-actions} shows  
trajectory rollouts sampled from this noise model (in a Gibson\cite{xia2018gibson} home).  As shown, identical action sequences can lead to vastly different final locations. We also added RGB and Depth sensor noises. 

1. **Collision Dynamics and ‘Sliding'**: In 2019, when the agent takes an action that results in a collision, the agent **slides** along the obstacle as opposed to stopping. This behavior is prevalent in video game engines as it allows for smooth human control; it is also enabled by default in MINOS, Deepmind Lab, AI2 THOR, and Gibson v1. We have found that this behavior enables `cheating' by learned agents -- the agents exploit this sliding mechanism to take an effective path that appears to travel **through non-navigable regions** of the environment (like walls). Such policies fail disastrously in the real world where the robot bump sensors  force a stop on contact with obstacles. To rectify this issue, we modify Habitat-Sim to disable sliding on collisions. 

1. **Multiple cosmetic/minor changes**: Change in robot embodiment/size, camera resolution, height and orientation, etc -- to match LoCoBot. 


## Task 2: ObjectNav

In ObjectNav, an agent is initialized at a random starting position and orientation in an unseen environment and asked to find an instance of an object category (‘find a chair’) by navigating to  it. A map of the environment is not provided and the agent must only use its sensory input to navigate. 

The agent is equipped with an RGB-D camera and a (noiseless) GPS+Compass sensor. GPS+Compass sensor provides the agent’s current location and orientation information relative to the start of the episode. We attempt to match the camera specification (field of view, resolution) in simulation to the Azure Kinect camera, but this task does not involve any sensing noise. 

We use 90 of the Matterport3D scenes (MP3D)~\cite{matterport} with the standard splits of train/val/test as prescribed by Anderson \etal \cite{anderson2018evaluation}. MP3D contains 40 annotated categories. We hand-select a subset of 21 by excluding categories that are not visually well defined like doorways or windows and architectural elements like walls, floors, and ceilings. 

We follow a similar evaluation protocol as prescribed/used by \cite{anderson2018evaluation, habitat19iccv, habitat_challenge} -- measuring **success** (did the agent navigate to the goal object?) and **efficiency** (how efficient was the path it took compared to an optimal path?). 


## Challenge Dataset

We create a set of PointGoal navigation episodes for the Gibson [[2]](#references) 3D scenes and a set of ObjectGoal navigation episodes for Matterport3D. We use the splits provided by the Gibson and Matterport3D dataset, retaining the train, and val sets, and separating the test set into test-standard and test-challenge. The train and val scenes are provided to participants. The test scenes are used for the official challenge evaluation and are not provided to participants.


## Evaluation

After calling the STOP action, the agent is evaluated using the "Success weighted by Path Length" (SPL) metric [[3]](#references).

<p align="center">
  <img src='res/img/spl.png' />
</p>

An episode is deemed successful if on calling the STOP action, the agent is within 0.36m of the Point goal position or with 1.0m Euclidean distance from an Object goal from target category that the object can be visible. The evaluation will be carried out in completely new houses which are not present in training and validation splits.

## Participation Guidelines

Participate in the contest by registering on the [EvalAI challenge page](https://evalai.cloudcv.org/web/challenges/challenge-page/254) and creating a team. Participants will upload docker containers with their agents that evaluated on a AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to make sure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation

1. Clone the challenge repository:  

    ```bash
    git clone https://github.com/facebookresearch/habitat-challenge.git
    cd habitat-challenge
    ```
    Implement your own agent or try one of ours. We provide hand-coded agents in `baselines/agents/simple_agents.py`, below is an example forward-only code for agent:
    ```python
    import habitat

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
    ```
    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to `GoalFollower` agent in `myagent/agent.py`.


1. Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) Note: only supports Linux; no Windows or MacOS.

1. Modify the provided Dockerfile if you need custom modifications. Let's say your code needs `pytorch`, these dependencies should be pip installed inside a conda environment called `habitat` that is shipped with our habitat-challenge docker, as shown below:

    ```dockerfile
    FROM fairembodied/habitat-challenge:2020

    ADD agent.py /agent.py
    ADD submission.sh /submission.sh
    ```
    Build your docker container: `docker build -t my_submission .` (Note: you will need `sudo` priviliges to run this command)

1. a) PoinGoal Navigation: Download Gibson scenes used for Habitat Challenge. Accept terms [here](https://docs.google.com/forms/d/e/1FAIpQLSen7LZXKVl_HuiePaFzG_0Boo6V3J5lJgzt3oPeSfPr4HTIEA/viewform) and select the download corresponding to “Habitat Challenge Data for Gibson (1.4 GB)“. Place this data in: `habitat-challenge/habitat-challenge-data/gibson`
   
   b) ObjectGoal Navigation: Download Matterport3D scenes used for Habitat Challenge [here](https://niessner.github.io/Matterport/). Place this data in: `habitat-challenge/habitat-challenge-data/mp3d`

1. Evaluate your docker container locally on RGB modality:
    ```bash
    ./test_locally_objectnav_rgbd.sh --docker-name my_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2019-02-14 21:23:51,798 initializing sim Sim-v0
    2019-02-14 21:23:52,820 initializing task Nav-v0
    2020-02-14 21:23:56,339 distance_to_goal: 5.205519378185272
    2020-02-14 21:23:56,339 spl: 0.0
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.**  
    To evaluate an agent for PointGoal challenge track run:
    ```bash
    ./test_locally_pointgoal_rgbd.sh --docker-name my_submission
    ```

### Online submission

Follow instructions in the `submit` tab of the [EvalAI challenge page](https://evalai.cloudcv.org/web/challenges/challenge-page/254) to submit your docker image. Note that you will need a version of EvalAI `>= 1.2.3`. Pasting those instructions here for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.2.3"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push my_submission:latest --phase <phase-name>
```

Valid challenge phases are `habitat20-{pointnav, objectnav}-{minival, test-std, test-ch}`.

The challenge consists of the following phases:

1. **Minival phase**: consists of 30 episodes with RGBD modality. This split is same as the one used in `./test_locally_{pointgoal, objectnav}_rgbd.sh`. The purpose of this phase is sanity checking -- to confirm that our remote evaluation reports the same result as the one you're seeing locally. Each team is allowed maximum of 30 submission per day for this phase, but please use them judiciously. We will block and disqualify teams that spam our servers. 
1. **Test Standard phase**: consists of episodes with RGBD modality. Each team is allowed maximum of 10 submission per day for this phase, but again, please use them judiciously. Don't overfit to the test set. The purpose of this phase is to serve as a public leaderboard establishing the state of the art. 
1. **Test Challenge phase**: consists of episodes with RGBD modality. This split will be used to decide challenge winners on the RGB track. Each team is allowed total of 5 submissions until the end of challenge submission phase. Results on this split will not be made public until the announcement of final results at the [Embodied AI workshop at CVPR](https://embodied-ai.org/). 

Note: Your agent will be evaluated on 1000-2000 episodes and will have a total available time of 30 mins to finish. Your submissions will be evaluated on AWS EC2 p2.xlarge instance which has a Tesla K80 GPU (12 GB Memory), 4 CPU cores, and 61 GB RAM. If you need more time/resources for evaluation of your submission please get in touch. If you face any issues or have questions you can ask them on the [habitat-challenge forum](https://evalai-forum.cloudcv.org/c/habitat-challenge-2020).


## Citing Habitat
Please cite the following paper for details about the 2019 challenge:

```
@inproceedings{habitat19iccv,
  title     =     {Habitat: {A} {P}latform for {E}mbodied {AI} {R}esearch},
  author    =     {Manolis Savva and Abhishek Kadian and Oleksandr Maksymets and Yili Zhao and Erik Wijmans and Bhavana Jain and Julian Straub and Jia Liu and Vladlen Koltun and Jitendra Malik and Devi Parikh and Dhruv Batra},
  booktitle =     {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      =     {2019}
}
```

## Acknowledgments

The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team and the data of [Gibson](http://gibsonenv.stanford.edu/) team.

## References

[1] [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[2] [Gibson env: Real-world perception for embodied agents](https://arxiv.org/abs/1808.10654). F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese. In CVPR, 2018

[3] [On evaluation of embodied navigation agents](https://arxiv.org/abs/1807.06757). Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, Amir R. Zamir. arXiv:1807.06757, 2018.
