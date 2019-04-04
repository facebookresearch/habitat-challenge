<p align="center">
  <img width = "50%" src='res/img/habitat_logo_with_text_horizontal_blue.png' />
  </p>

--------------------------------------------------------------------------------

# Habitat-Challenge

This repository contains starter code for the challenge and details of task, training and evaluation. For an overview of habitat-challenge visit [aihabitat.org/challenge](https://aihabitat.org/challenge).

## Task

The objective of the agent is to navigate successfully to a target location specified by agent-relative Euclidean coordinates 
(e.g. "Go 5m north, 3m west relative to current location"). Importantly, updated goal-specification (relative coordinates) is provided at all times (as the episode progresses) and 
not just at the outset of an episode. The action space for the agent consists of turn-left, turn-right, move forward and STOP actions. The turn actions produce a turn of 10 degrees and move forward action produces a linear displacement of 0.25m. The STOP action is used by the agent to indicate completion of an episode. We use an idealized embodied agent with a cylindrical body with diameter 0.2m and height 1.5m. The challenge consists of two tracks, RGB (only RGB input) and RGBD (RGB and Depth inputs).

## Challenge Dataset

We create a set of PointGoal navigation episodes for the Gibson [1] 3D scenes as the main dataset for the challenge. Gibson was preferred over SUNCG because unlike SUNCG it contains scans of real-world indoor environments. Gibson was chosen over Matterport3D because unlike Matterport3D Gibson's raw meshes are not publicly available allowing us to sequester a test set. We use the splits provided by the Gibson dataset, retaining the train, and val sets, and separating the test set into test-standard and test-challenge. The train and val scenes are provided to participants. The test scenes are used for the official challenge evaluation and are not provided to participants. 


## Evaluation

After calling the STOP action, the agent is evaluated using the "Success weighted by Path Length" (SPL) metric [1].

<p align="center">
  <img src='res/img/spl.png' />
</p>

An episode is deemed successful if on calling the STOP action, the agent is within 0.2m of the goal position. The evaluation will be carried out in completely new houses which are not present in training and validation splits.

## Participation Guidelines

Participate in the contest by registering on the [EvalAI challenge page](https://evalai.cloudcv.org/web/challenges/challenge-page/254) and creating a team. Participants will upload docker containers with their agents that evaluated on a AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to make sure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation

1. Clone the challenge repository:  

    ```bash
    git clone https://github.com/facebookresearch/habitat-challenge.git
    cd habitat-challenge-staging
    ```
   Implement your own agent or try one of ours. We provide an example hand-coded agent in `myagent/agent_forwardonly.py`, below is the code for agent:

    ```python
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
    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to myagent/agent_forwardonly.py.
                

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) Note: only supports Linux; no Windows or MacOS.

1. Modify the provided Dockerfile if you need custom modifications. Let's say your code needs `pytorch`, these dependencies should be pip installed inside a conda environment called `habitat` that is shipped with our habitat-challenge docker, as shown below:

    ```dockerfile
    FROM fairembodied/habitat-challenge:latest
    
    # install dependencies in the habitat conda environment
    RUN /bin/bash -c ". activate habitat; pip install torch"
    
    ADD myagent /myagent
    ADD submission.sh /submission.sh
    ```
    Build your docker container: `docker build -t my_submission .` (Note: you will need `sudo` priviliges to run this command)
    
1. Download Gibson scenes used for Habitat Challenge. Accept terms [here](https://docs.google.com/forms/d/e/1FAIpQLSen7LZXKVl_HuiePaFzG_0Boo6V3J5lJgzt3oPeSfPr4HTIEA/viewform) and select the download corresponding to “Habit    at Challenge Data for Gibson (1.4 GB)“. Place this data in: `habitat-challenge/habitat-challenge-data/gibson`

1. Evaluate your docker container locally:
    ```bash
    ./test_locally.sh --docker-name my_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2019-03-17 22:35:10,692 initializing sim Sim-v0
    2019-03-17 22:35:11,844 initializing task Nav-v0
    2019-03-17 22:35:30,979 Time taken for evaluation of agent: 19.135s
    2019-03-17 22:35:30,980 spl: 0.0
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.**

### Online submission

Follow instructions in the `submit` tab of the [EvalAI challenge page](https://evalai.cloudcv.org/web/challenges/challenge-page/254) to submit your docker image. Pasting those instructions here for convenience: 

```bash
# Installing EvalAI Command Line Interface
pip install evalai
evalai set_token <your EvalAI participant token>
evalai push my_submission:latest --phase habitat19-rgb-val
```

Valid challenge phases are `habitat19-{rgb, rgbd}-{val, test-std, test-chal}`. 

Note: Your agent will be evaluated on 1000 episodes and will have a total available time of 30mins to finish. Your submissions will be evaluated on AWS EC2 p2.xlarge instance which has a Tesla K80 GPU (12 GB Memory), 4 CPU cores, and 61 GB RAM. 

### Starter code and Training

1. Install the [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/) and [Habitat-API](https://github.com/facebookresearch/habitat-api/) packages.

1. Download the Gibson dataset following the instructions [here](https://github.com/StanfordVL/GibsonEnv#database). After downloading extract the dataset to folder `habitat-api/data/scene_datasets/gibson/` folder (this folder should contain the `.glb` files from gibson). Note that the `habitat-api` folder is the [habitat-api](https://github.com/facebookresearch/habitat-api/) repository folder.

1. Download the dataset for Gibson pointnav from [link](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) and place it in the folder `habitat-api/data/datasets/pointnav/gibson`. If placed correctly, you should have the train and val splits at `habitat-api/data/datasets/pointnav/gibson/v1/train/` and `habitat-api/data/datasets/pointnav/gibson/v1/val/` respectively. Place Gibson scenes downloaded in step-4 of local-evaluation under the `habitat-api/data/scene_datasets` folder.

1. An example PPO baseline for the Pointnav task is present at [`habitat-api/baselines`](https://github.com/facebookresearch/habitat-api/tree/master/baselines). To start training on the Gibson dataset using this implementation follow instructions in the [baselines README](https://github.com/fairinternal/habitat-api/tree/master/baselines#baselines). Set `--task-config` to `tasks/pointnav_gibson.yaml` to train on Gibson data. This is a good starting point for participants to start building their own models. The PPO implementation contains initialization and interaction with the environment as well as tracking of training statistics. Participants can borrow the basic blocks from this implementation to start building their own models.

1. To evaluate a trained PPO model on Gibson val split run `evaluate_ppo.py` using instructions in the README at [`habitat-api/baselines`](https://github.com/facebookresearch/habitat-api/tree/master/baselines) with `--task-config` set to `tasks/pointnav_gibson.yaml`. The evaluation script will report SPL metric.

1. You can also use the general benchmarking script present at [`benchmark.py`](https://github.com/facebookresearch/habitat-api/blob/master/examples/benchmark.py). Set `--task-config` to `tasks/pointnav_gibson.yaml`  and inside the file [`tasks/pointnav_gibson.yaml`](https://github.com/facebookresearch/habitat-api/blob/master/configs/tasks/pointnav_gibson.yaml) set the `SPLIT` to `val`.

1. Once you have trained your agents you can follow the submission instructions above to test them locally as well as submit them for online evaluation.

## Acknowledgments

The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team and the data of [Gibson](http://gibsonenv.stanford.edu/) team. We are especially grateful to Rishabh Jain, Deshraj Yadav, Fei Xia and Amir Zamir.

## References

[1] F. Xia,  A. R. Zamir,  Z. He,  A. Sax,  J. Malik,  and S. Savarese,  "Gibson env: Real-world perception for embodied agents," in CVPR, 2018
