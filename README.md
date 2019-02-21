<p align="center">
  <img width = "50%" src='res/img/habitat_logo_with_text_horizontal_blue.png' />
  </p>

--------------------------------------------------------------------------------

# Habitat-Challenge

Habitat Challenge is an autonomous navigation challenge that aims to benchmark and accelerate progress in embodied AI. In its first iteration, Habitat Challenge 2019 is based on the PointGoal task as defined by Anderson et al. [1]. 

To participate in the challenge follow the instructions at [link](https://evalai.cloudcv.org/web/challenges/challenge-page/230/overview). The challenge is built on top of the [habitat](https://aihabitat.org) stack. The winners of challenge will be announced at [CVPR-2019](http://cvpr2019.thecvf.com/) at the [**Habitat: Embodied Agents Challenge and Workshop**](http://cvpr2019.thecvf.com/program/workshops). Details on the prizes coming soon. If you are interested in participating please fill out your name and email address in this [form](https://goo.gl/forms/LpbZ8p8wv4imXmB03) and we will get back to you when the submission phase goes live.

# Background

One fundamental difficulty in creating a challenge around embodied AI tasks is the transition from static predictions (as in supervised learning) to sequential decision making (as in reinforcement learning). 
In classical 'internet AI' challenges (e.g. ImageNet, COCO, VQA), it is possible to release a static testing 
dataset and ask participants to simply upload their predictions on this set. Embodied AI tasks (navigation, instruction following, question answering) typically involve sequential decision making and agent-driven 
control, making it infeasible to pre-package a testing dataset. 
Essentially, embodied AI challenges require participants 
to _upload code not predictions_. The uploaded agents may then be evaluated in novel (unseen) test environments. 

We leverage the frontend and challenge submission process 
of the [EvalAI platform](https://evalai.cloudcv.org/), and build backend infrastructure ourselves. 
Participants in Habitat Challenge are asked to upload (on EvalAI) Docker containers with their 
agents that are then evaluated on a live AWS GPU-enabled instance. 

### Participation and Starter Code

The challenge is hosted on EvalAI, to participate please refer to the [instructions](https://evalai.cloudcv.org/web/challenges/challenge-page/230/overview). To get started:

1. Install the [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/) and [Habitat-API](https://github.com/facebookresearch/habitat-api/) packages.
2. Download the Gibson dataset following the instructions [here](https://github.com/StanfordVL/GibsonEnv#database). After downloading extract the dataset to folder `habitat-api/data/scene_datasets/gibson/` folder (this folder should contain the `.glb` files from gibson). Note that the `habitat-api` folder is the [habitat-api](https://github.com/facebookresearch/habitat-api/) repository folder.
3. Download the dataset for Gibson pointnav from [link](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) and place it in the folder `habitat-api/data/datasets/pointnav/gibson`. If placed correctly, you should have the train and val splits at `habitat-api/data/datasets/pointnav/gibson/v1/train/` and `habitat-api/data/datasets/pointnav/gibson/v1/val/` respectively.
4. An example PPO baseline for the Pointnav task is present at [`habitat-api/baselines`](https://github.com/facebookresearch/habitat-api/tree/master/baselines). To start training on the Gibson dataset using this implementation follow instructions in the README. Set `--task-config` to `tasks/pointnav_gibson.yaml` to train on Gibson data. This is a good starting point for participants to start building their own models. The PPO implementation contains initialization and interaction with the environment as well as tracking of training statistics. Participants can borrow the basic blocks from this implementation to start building their own models.
5. To evaluate a trained PPO model on Gibson val split run `evaluate_ppo.py` using instructions in the README at [`habitat-api/baselines`](https://github.com/facebookresearch/habitat-api/tree/master/baselines) with `--task-config` set to `tasks/pointnav_gibson.yaml`. The evaluation script will report SPL metric.
6. You can also use the general benchmarking script present at [`benchmark.py`](https://github.com/facebookresearch/habitat-api/blob/master/examples/benchmark.py). Set `--task-config` to `tasks/pointnav_gibson.yaml`  and inside the file [`tasks/pointnav_gibson.yaml`](https://github.com/facebookresearch/habitat-api/blob/master/configs/tasks/pointnav_gibson.yaml) set the `SPLIT` to `val`.
7. Instructions for submission will be made available soon when our challenge submission infrastructure goes live.

### Task

The objective of the agent is to navigate successfully to a target location specified by agent-relative Euclidean coordinates 
(e.g. "Go 5m north, 3m west relative to current location"). 
Importantly, updated goal-specification (relative coordinates) is provided at all times (as the episode progresses) and 
not just at the outset of an episode. 
The action space for the agent consists of turn-left, turn-right, move forward and STOP actions. The turn actions produce a turn of 10 degrees and move forward action produces a linear displacement of 0.25m. The STOP action is used by the agent to indicate completion of an episode. We use an idealized embodied agent with a cylindrical body with diameter 0.2m and height 1.5m. The challenge consists of two tracks, RGB (only RGB input) and RGBD (RGB and Depth inputs).

### Challenge Dataset

We create a set of PointGoal navigation episodes for the Gibson [2] 3D scenes as the main dataset for the challenge. 
Gibson was preferred over SUNCG because unlike SUNCG it contains scans of real-world indoor environments. Gibson was chosen 
over Matterport3D because unlike Matterport3D Gibson's raw meshes are not publicly available allowing us to sequester a test set. We use the splits provided by the Gibson dataset, retaining the train, and val sets, and separating the test set into test-standard and test-challenge. The train and val scenes are provided to participants. The test scenes are used for the official challenge evaluation and are not  provided to participants. 


### Evaluation

After calling the STOP action, the agent is evaluated using the "Success weighted by Path Length" (SPL) metric [1].

<p align="center">
  <img src='res/img/spl.png' />
</p>

An episode is deemed successful if on calling the STOP action, the agent is within 0.2m of the goal position. The evaluation will be carried out in completely new houses which are not present in training and validation splits.

### Submission Instructions

Coming soon.

### Acknowledgments

The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team and the data of [Gibson](http://gibsonenv.stanford.edu/) team. We are especially grateful to Rishabh Jain, Deshraj Yadav, Fei Xia and Amir Zamir.

## License

Habitat-Challenge is MIT licensed. See the LICENSE file for details.

## References

[1] P. Anderson, A. X. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva, and A. R. Zamir, "On evaluation of embodied navigation agents," arXiv:1807.06757, 2018

[2] F. Xia,  A. R. Zamir,  Z. He,  A. Sax,  J. Malik,  and S. Savarese,  "Gibson env: Real-world perception for embodied agents," in CVPR, 2018
