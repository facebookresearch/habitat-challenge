<p align="center">
  <img width = "50%" src='res/img/habitat_logo_with_text_horizontal_blue.png' />
  </p>

--------------------------------------------------------------------------------

# Habitat-Challenge

This repository contains starter code for the 2020 challenge, details of the tasks, and training and evaluation setups. For an overview of habitat-challenge visit [aihabitat.org/challenge](https://aihabitat.org/challenge/).

This year, we are hosting challenges on two embodied navigation tasks: 
1. PointNav (*‘Go 5m north, 3m west relative to start’*)
1. ObjectNav (*‘find a chair’*). 

Task #1: PointNav focuses on realism and *sim2real predictivity* (the ability to predict the performance of a nav-model on a real robot from its performance in simulation). 

Task #2: ObjectNav focuses on egocentric object/scene recognition and a commonsense understanding of object semantics (where is a fireplace typically located in a house?). 


## Task 1: PointNav 
In PointNav, an agent is spawned at a random starting position and orientation in an unseen environment and and asked to navigate to target coordinates specified relative to the agent’s start location (*‘Go 5m north, 3m west relative to start’*). No ground-truth map is available and the agent must only use its sensory input (an RGB-D camera) to navigate. 

### Dataset
We use [Gibson 3D scenes](http://gibsonenv.stanford.edu/database/) for the challenge. As in the 2019 Habitat challenge, we use the splits provided by the Gibson dataset, retaining the train and val sets, and separating the test set into test-standard and test-challenge. The train and val scenes are provided to participants. The test scenes are used for the official challenge evaluation and are not provided to participants. Note: The agent size has changed from 2019, thus the navigation episodes have changed (a wider agent in 2020 rendered many of 2019 episodes unnavigable). 

### Evaluation
After calling the STOP action, the agent is evaluated using the 'Success weighted by Path Length' (SPL) metric [2]. 

<p align="center">
  <img src='res/img/spl.png' />
</p>


An episode is deemed successful if on calling the STOP action, the agent is within 0.36m (2x agent-radius) of the goal position.


### New in 2020
The main emphasis in 2020 is on increased realism and on sim2real predictivity (the ability to predict performance on a real robot from its performance in simulation). 

Specifically, we introduce the following changes inspired by our experiments and findings in [3]: 

1. **No GPS+Compass sensor**: In 2019, the relative coordinates specifying the goal were continuously updated during agent movement &mdash; essentially simulating an agent with perfect localization and heading estimation (*e.g.* an agent with an idealized GPS+Compass). However, high-precision localization in indoor environments can not be assumed in realistic settings &mdash; GPS has low precision indoors, (visual) odometry may be noisy, SLAM-based localization can fail, etc. Hence, in 2020's challenge the agent does NOT have a GPS+Compass sensor and must navigate solely using an egocentric RGB-D camera. This change elevates the need to perform RGBD-based online localization. 

1. **Noisy Actuation and Sensing**: In 2019, the agent actions were deterministic &mdash; *i.e.* when the agent executes *turn-left 30 degrees*, it turns *exactly* 30 degrees, and forward 0.25 m moves the agent *exactly* 0.25 m forward (modulo collisions). However, no robot moves deterministically &mdash; actuation error, surface properties such as friction, and a myriad of other sources of error introduce significant drift over a long trajectory. To model this, we introduce a noise model acquired by benchmarking the [Locobot](http://www.locobot.org/) robot by the [PyRobot](https://www.pyrobot.org/) team. We also added RGB and Depth sensor noises. 

    <p align="center">
      <img src="res/img/pyrobot-noise-rollout.png" height="200"/>
    </p>
    Figure shows the effect of actuation noise. The black line is the trajectory of an action sequence with perfect actuation (no noise). In red are multiple rollouts of this action sequence sampled from the actuation noise model. As we can see, identical action sequences can lead to vastly different final locations.

1. **Collision Dynamics and ‘Sliding'**: In 2019, when the agent takes an action that results in a collision, the agent *slides* along the obstacle as opposed to stopping. This behavior is prevalent in video game engines as it allows for smooth human control; it is also enabled by default in MINOS, Deepmind Lab, AI2 THOR, and Gibson v1. We have found that this behavior enables 'cheating' by learned agents &mdash; the agents exploit this sliding mechanism to take an effective path that appears to travel *through non-navigable regions* of the environment (like walls). Such policies fail disastrously in the real world where the robot bump sensors  force a stop on contact with obstacles. To rectify this issue, we modify [Habitat-Sim to disable sliding on collisions](https://github.com/facebookresearch/habitat-sim/pull/439). 

1. **Multiple cosmetic/minor changes**: Change in robot embodiment/size, camera resolution, height, and orientation, etc &mdash; to match LoCoBot. 


## Task 2: ObjectNav

In ObjectNav, an agent is initialized at a random starting position and orientation in an unseen environment and asked to find an instance of an object category (*‘find a chair’*) by navigating to  it. A map of the environment is not provided and the agent must only use its sensory input to navigate. 

The agent is equipped with an RGB-D camera and a (noiseless) GPS+Compass sensor. GPS+Compass sensor provides the agent’s current location and orientation information relative to the start of the episode. We attempt to match the camera specification (field of view, resolution) in simulation to the Azure Kinect camera, but this task does not involve any injected sensing noise. 

### Dataset
We use 90 of the [Matterport3D scenes (MP3D)](https://niessner.github.io/Matterport/) with the standard splits of train/val/test as prescribed by Anderson *et al.* [2]. MP3D contains 40 annotated categories. We hand-select a subset of 21 by excluding categories that are not visually well defined (like doorways or windows) and architectural elements (like walls, floors, and ceilings). 

### Evaluation
We generalize the PointNav evaluation protocol used by [1,2,3] to ObjectNav. At a high-level, we measure performance along the same two axes:  
- **Success**: Did the agent navigate to an instance of the goal object? (Notice: *any* instance, regardless of distance from starting location.)
- **Efficiency**: How efficient was the agent's path compared to an optimal path? (Notice: optimal path = shortest path from the agent's starting position to the *closest* instance of the target object category.)

Concretely, an episode is deemed successful if on calling the STOP action, the agent is within 1.0m Euclidean distance from any instance of the target object category AND the object *can be viewed by an oracle* from that stopping position by turning the agent or looking up/down. Notice: we do NOT require the agent to be actually viewing the object at the stopping location, simply that the such oracle-visibility is possible without moving. Why? Because we want participants to focus on *navigation* not object framing. In the larger goal of Embodied AI, the agent is navigating to an object instance in order to interact with is (say point at or manipulate an object). Oracle-visibility is our proxy for *'the agent is close enough to interact with the object'*. 

ObjectNav-SPL is defined analogous to PointNav-SPL. The only key difference is that the shortest path is computed to the object instance closest to the agent start location. Thus, if an agent spawns very close to *'chair1'* but stops at a distant *'chair2'*, it will be achieve 100% success (because it found a *'chair'*) but a fairly low SPL (because the agent path is much longer compared to the oracle path). 

## Participation Guidelines

Participate in the contest by registering on the [EvalAI challenge page](https://evalai.cloudcv.org/web/challenges/challenge-page/254) and creating a team. Participants will upload docker containers with their agents that evaluated on a AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to make sure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation

1. Clone the challenge repository:  

    ```bash
    git clone https://github.com/facebookresearch/habitat-challenge.git
    cd habitat-challenge
    ```

1. Implement your own agent or try one of ours. We provide hand-coded agents in `baselines/agents/simple_agents.py`; below is a code-snippet for an agent that takes random actions:
    ```python
    import habitat

    class RandomAgent(habitat.Agent):
        def reset(self):
            pass

        def act(self, observations):
            return {"action": numpy.random.choice(task_config.TASK.POSSIBLE_ACTIONS)}

    def main():
        agent = RandomAgent(task_config=config)
        challenge = habitat.Challenge()
        challenge.submit(agent)
    ```
    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to `RandomAgent` agent in `agent.py`. 


1. Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following instructions here: [https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)). 
Note: only supports Linux; no Windows or MacOS.

1. Modify the provided Dockerfile if you need custom modifications. Let's say your code needs `pytorch`, these dependencies should be pip installed inside a conda environment called `habitat` that is shipped with our habitat-challenge docker, as shown below:

    ```dockerfile
    FROM fairembodied/habitat-challenge:2020

    # install dependencies in the habitat conda environment
    RUN /bin/bash -c ". activate habitat; pip install torch"

    ADD agent.py /agent.py
    ADD submission.sh /submission.sh
    ```
    Build your docker container: `docker build . -t my_submission` where `my_submission` is the docker image name you want to use. (Note: you may need `sudo` priviliges to run this command.)

1. a) PoinNav: Download Gibson scenes used for Habitat Challenge. Accept terms [here](https://docs.google.com/forms/d/e/1FAIpQLSen7LZXKVl_HuiePaFzG_0Boo6V3J5lJgzt3oPeSfPr4HTIEA/viewform) and select the download corresponding to “Habitat Challenge Data for Gibson (1.5 GB)“. Place this data in: `habitat-challenge/habitat-challenge-data/data/scene_datasets/gibson`
   
   b) ObjectNav: Download Matterport3D scenes used for Habitat Challenge [here](https://niessner.github.io/Matterport/). Place this data in: `habitat-challenge/habitat-challenge-data/data/scene_datasets/mp3d`

1. Evaluate your docker container locally:
    ```bash
    # Testing PointNav
    ./test_locally_pointnav_rgbd.sh --docker-name my_submission
    
    # Testing ObjectNav
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

### Online submission

Follow instructions in the `submit` tab of the EvalAI challenge page (coming soon) to submit your docker image. Note that you will need a version of EvalAI `>= 1.2.3`. Pasting those instructions here for convenience:

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

1. **Minival phase**: consists of 30 episodes with RGBD modality. This split is same as the one used in `./test_locally_{pointnav, objectnav}_rgbd.sh`. The purpose of this phase is sanity checking -- to confirm that our remote evaluation reports the same result as the one you're seeing locally. Each team is allowed maximum of 30 submission per day for this phase, but please use them judiciously. We will block and disqualify teams that spam our servers. 
1. **Test Standard phase**: consists of episodes with RGBD modality. Each team is allowed maximum of 10 submission per day for this phase, but again, please use them judiciously. Don't overfit to the test set. The purpose of this phase is to serve as a public leaderboard establishing the state of the art. 
1. **Test Challenge phase**: consists of episodes with RGBD modality. This split will be used to decide challenge winners on the RGB track. Each team is allowed total of 5 submissions until the end of challenge submission phase. Results on this split will not be made public until the announcement of final results at the [Embodied AI workshop at CVPR](https://embodied-ai.org/). 

Note: Your agent will be evaluated on 1000-2000 episodes and will have a total available time of 30 mins to finish. Your submissions will be evaluated on AWS EC2 p2.xlarge instance which has a Tesla K80 GPU (12 GB Memory), 4 CPU cores, and 61 GB RAM. If you need more time/resources for evaluation of your submission please get in touch. If you face any issues or have questions you can ask them on the habitat-challenge forum (coming soon).


## Citing Habitat Challenge 2020
Please cite the following paper for details about the 2020 PointNav challenge:

```
@inproceedings{habitat2020sim2real,
  title     =     {Are We Making Real Progress in Simulated Environments? Measuring the Sim2Real Gap in Embodied Visual Navigation},
  author    =     {Abhishek Kadian, Joanne Truong, Aaron Gokaslan, Alexander Clegg, Erik Wijmans, Stefan Lee, Manolis Savva, Sonia Chernova, Dhruv Batra},
  booktitle =     {arXiv:1912.06321},
  year      =     {2019}
}
```

## Acknowledgments

The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team. We also thank the work behind [Gibson](http://gibsonenv.stanford.edu/) and [Matterport3D](https://niessner.github.io/Matterport/) datasets. 

## References

[1] [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Manolis Savva\*, Abhishek Kadian\*, Oleksandr Maksymets\*, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[2] [On evaluation of embodied navigation agents](https://arxiv.org/abs/1807.06757). Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, Amir R. Zamir. arXiv:1807.06757, 2018.

[3] [Are We Making Real Progress in Simulated Environments? Measuring the Sim2Real Gap in Embodied Visual Navigation](https://abhiskk.github.io/sim2real). Abhishek Kadian\*, Joanne Truong\*, Aaron Gokaslan, Alexander Clegg, Erik Wijmans, Stefan Lee, Manolis Savva, Sonia Chernova, Dhruv Batra. arXiv:1912.06321, 2019.
