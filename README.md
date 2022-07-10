<p align="center">
  <img width = "50%" src='res/img/habitat_logo_with_text_horizontal_blue.png' />
  </p>

--------------------------------------------------------------------------------

# Habitat Rearrange Challenge 2022

This repository contains the starter code for the 2022 challenge, and training and evaluation setups. For an overview of habitat-challenge, visit [aihabitat.org/challenge/rearrange_2022](https://aihabitat.org/challenge/rearrange_2022).

## Task: Object Rearrangement

In ObjectNav, an agent is initialized at a random starting position and orientation in an unseen environment and asked to find an instance of an object category (*‘find a chair’*) by navigating to it. A map of the environment is not provided and the agent must only use its sensory input to navigate.

The agent is equipped with an RGB-D camera and a (noiseless) GPS+Compass sensor. GPS+Compass sensor provides the agent’s current location and orientation information relative to the start of the episode. We attempt to match the camera specification (field of view, resolution) in simulation to the Azure Kinect camera, but this task does not involve any injected sensing noise.

For details about the agent, dataset, and evaluation, see the challenge website: [aihabitat.org/challenge/rearrange_2022](https://aihabitat.org/challenge/rearrange_2022).

## Participation Guidelines

Participate in the contest by registering on the [EvalAI challenge page](https://eval.ai/web/challenges/challenge-page/1615/overview) and creating a team. Participants will upload docker containers with their agents that are evaluated on an AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to ensure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation

1. Clone the challenge repository:

    ```bash
    git clone -b rearrangement-challenge-2022 https://github.com/facebookresearch/habitat-challenge.git
    cd habitat-challenge
    ```

1. Implement your own agent or try one of ours. We provide an agent in `agents/random_agent.py` that takes random actions.

    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to `RandomAgent` agent in `agent.py`.


1. Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following instructions here: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
Note: only supports Linux; no Windows or MacOS.

1. Modify the provided Dockerfile if you need custom modifications. Let’s say your code needs `pytorch==1.9.0`, these dependencies should be pip installed inside a conda environment called `habitat` that is shipped with our habitat-challenge docker, as shown below:

    ```dockerfile
    FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

    # install dependencies in the habitat conda environment
    RUN /bin/bash -c ". activate habitat; pip install torch==1.9.0"

    ADD agent.py /agent.py
    ADD submission.sh /submission.sh
    ```
    Build your docker container using: 
    ```bash
    docker build . --file docker/hab2.Dockerfile  -t rearrange_submission
    ```

    Note #1: you may need `sudo` priviliges to run this command.

    Note #2: Please make sure that you keep your local version of `fairembodied/habitat-challenge:testing_2022_habitat_base_docker` image up to date with the image we have hosted on [dockerhub](https://hub.docker.com/r/fairembodied/habitat-challenge/tags). This can be done by pruning all cached images, using:
    ```
    docker system prune -a
    ```

1. Dataset: Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/) `Habitat-Sim` via these instructions and download the dataset with `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`.

1. Evaluate your docker container locally:
    ```bash
    ./test_locally_objectnav_rgbd.sh --docker-name objectnav_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2022-02-14 01:23:51,798 initializing sim Sim-v0
    2022-02-14 01:23:52,820 initializing task Nav-v0
    2022-02-14 01:23:56,339 distance_to_goal: 5.205519378185272
    2022-02-14 01:23:56,339 spl: 0.0
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.**

### Online submission

Follow instructions in the `submit` tab of the [EvalAI challenge page](https://eval.ai/web/challenges/challenge-page/1615/submission) to submit your docker image. Note that you will need a version of EvalAI `>= 1.3.13`. Pasting those instructions here for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.3.13"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push objectnav_submission:latest --phase <phase-name>
```

Valid phase names are `habitat-objectnav-{minival, test-standard, test-challenge}-2022-1615`. The challenge consists of the following phases:

1. **Minival phase**: This split is same as the one used in `./test_locally_objectnav_rgbd.sh`. The purpose of this phase/split is sanity checking -- to confirm that our remote evaluation reports the same result as the one you’re seeing locally. Each team is allowed maximum of 100 submissions per day for this phase, but please use them judiciously. We will block and disqualify teams that spam our servers.
1. **Test Standard phase**: The purpose of this phase/split is to serve as the public leaderboard establishing the state of the art; this is what should be used to report results in papers. Each team is allowed maximum of 10 submissions per day for this phase, but again, please use them judiciously. Don’t overfit to the test set.
1. **Test Challenge phase**: This phase/split will be used to decide challenge winners. Each team is allowed a total of 5 submissions until the end of challenge submission phase. The highest performing of these 5 will be automatically chosen. Results on this split will not be made public until the announcement of final results at the [Embodied AI workshop at CVPR](https://embodied-ai.org/).

Note: Your agent will be evaluated on 1000 episodes and will have a total available time of 48 hours to finish. Your submissions will be evaluated on AWS EC2 p3.2xlarge instance which has a V100 GPU (16 GB Memory), 8 vCPU cores, and 61 GB RAM. If you need more time/resources for evaluation of your submission please get in touch. If you face any issues or have questions you can ask them by opening an issue on this repository.


### Installing Habitat-Sim

1. Prepare your [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) env:
    ```bash
    # We require python>=3.7 and cmake>=3.10
    conda create -n habitat python=3.7 cmake=3.14.0
    conda activate habitat
    ```

1. Install Habitat-Sim using our custom Conda package for habitat challenge 2022 with: 
    ```
    conda install habitat-sim-challenge-2022 -c conda-forge -c aihabitat
    ```
    In case you face any issues related to the `GLIBCXX` version after conda installation, please uninstall this conda package and install the habitat-sim repository from source (more information [here](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source)). Make sure that you are using the `challenge-2022` tag and not the `stable` branch for your installation.

### PPO Starter Code
TODO

### DD-PPO Training Starter Code
Evaluate a [Habitat Baselines config](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines/config/rearrange). In this example, we will evaluate a DD-PPO baseline from Habitat Lab.
Follow these next steps to get the DD-PPO baseline running (skip to step 3 if you have completed step 5.b from the local evaluation section):

1. Install Habitat-Sim via these instructions.

1. Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/) - We have created the `challenge-tasks` tag in our Github repo, which can be cloned using: 
    ```
    git clone --branch challenge-tasks https://github.com/facebookresearch/habitat-lab.git
    ``` 
    Also ensure that habitat-baselines is installed when installing Habitat-Lab by using `python setup.py develop --all` . You will find further information for installation in the Github repositories. 

1. Download the dataset `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`

    If placed correctly, you should see the train, val and val mini splits in the `data/datasets/replica_cad/rearrange/v1/{train, val, minival}` folders respectively. 

1. An example on how to train DD-PPO model can be found in [habitat-lab/habitat_baselines/rl/ddppo](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines/rl/ddppo). See the corresponding README in habitat-lab for how to adjust the various hyperparameters, save locations, visual encoders and other features.

1. Follow this documentation for how to run DD-PPO in a single or multi-machine setup. 

    1. To run on a single machine use the following script from `habitat-lab` directory:
        ```bash
        #/bin/bash

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet
        export HABITAT_SIM_LOG=quiet

        set -x
        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node 1 \
            habitat_baselines/run.py \
            --exp-config ../habitat-challenge/configs/methods/ddppo_monolithic.yaml \
            --run-type train \
            BASE_TASK_CONFIG_PATH ../habitat-challenge/configs/rearrange.local.rgbd.yaml \
            TASK_CONFIG.DATASET.DATA_PATH ../habitat-challenge/data/replica_cad/v1/{split}/rearrange.json.gz \
            TASK_CONFIG.DATASET.SCENES_DIR ../habitat-challenge/data/replica_cad/ \
            TASK_CONFIG.DATASET.SPLIT 'train' \
            TENSORBOARD_DIR ./tb \
            CHECKPOINT_FOLDER ./checkpoints \
            LOG_FILE ./train.log
        ```
    1. There is also an example of running the code distributed on a cluster with SLURM. While this is not necessary, if you have access to a cluster, it can significantly speed up training. To run on multiple machines in a SLURM cluster run the following script: change `#SBATCH --nodes $NUM_OF_MACHINES` to the number of machines and `#SBATCH --ntasks-per-node $NUM_OF_GPUS` and `$SBATCH --gres $NUM_OF_GPUS` to specify the number of GPUS to use per requested machine.
        ```bash
        #!/bin/bash
        #SBATCH --job-name=ddppo
        #SBATCH --output=logs.ddppo.out
        #SBATCH --error=logs.ddppo.err
        #SBATCH --gres gpu:1
        #SBATCH --nodes 1
        #SBATCH --cpus-per-task 10
        #SBATCH --ntasks-per-node 1
        #SBATCH --mem=60GB
        #SBATCH --time=12:00
        #SBATCH --signal=USR1@600
        #SBATCH --partition=dev

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet

        export MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

        set -x
        srun python -u -m habitat_baselines.run \
            --exp-config ../habitat-challenge/configs/ddppo_objectnav.yaml \
            --run-type train \
            BASE_TASK_CONFIG_PATH ../habitat-challenge/configs/rearrange.local.rgbd.yaml \
            TASK_CONFIG.DATASET.DATA_PATH ../habitat-challenge/data/replica_cad/v1/{split}/rearrange.json.gz \
            TASK_CONFIG.DATASET.SCENES_DIR ../habitat-challenge/data/replica_cad/ \
            TASK_CONFIG.DATASET.SPLIT 'train' \
            TENSORBOARD_DIR ./tb \
            CHECKPOINT_FOLDER ./checkpoints \
            LOG_FILE ./train.log
        ```

    1. The preceding two scripts are based off ones found in the [habitat_baselines/ddppo](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines/rl/ddppo).

1. The checkpoint specified by `$PATH_TO_CHECKPOINT` can evaluated displaying videos and evaluation metrics:
    ```bash
    python -u -m habitat_baselines.run \
        --exp-config ../habitat-challenge/configs/ddppo_objectnav.yaml \
        --run-type eval \
        BASE_TASK_CONFIG_PATH ../habitat-challenge/configs/challenge_objectnav2022.local.rgbd.yaml \
        TASK_CONFIG.DATASET.DATA_PATH ../habitat-challenge/habitat-challenge-data/objectgoal_hm3d/{split}/{split}.json.gz \
        TASK_CONFIG.DATASET.SCENES_DIR ../habitat-challenge/habitat-challenge-data/data/scene_datasets/ \
        EVAL_CKPT_PATH_DIR $PATH_TO_CHECKPOINT \
        TASK_CONFIG.DATASET.SPLIT val
    ```
1. We provide Dockerfiles ready to use with the DD-PPO baselines in `hab2_DDPPO_baseline.Dockerfile`. For the sake of completeness, we describe how you can make your own Dockerfile below. If you just want to test the baseline code, feel free to skip this bullet because  `hab2_DDPPO_baseline.Dockerfile` is ready to use.
    1. You may want to modify the `hab2_DDPPO_baseline.Dockerfile` to include torchvision or other libraries. To install torchvision, ifcfg and tensorboard, add the following command to the Docker file:
        ```dockerfile
        RUN /bin/bash -c ". activate habitat; pip install ifcfg torchvision tensorboard"
        ```
    1. You change which `agent.py` and which `submission.sh` script is used in the Docker, modify the following lines and replace the first agent.py or submission.sh with your new files.:
        ```dockerfile
        ADD agent.py agent.py
        ADD submission.sh submission.sh
        ```
    1. Do not forget to add any other files you may need in the Docker, for example, we add the `demo.ckpt.pth` file which is the saved weights from the DD-PPO example code.

    1. Finally, modify the submission.sh script to run the appropriate command to test your agents. The scaffold for this code can be found `agent.py` and the DD-PPO specific agent can be found in `ddppo_agents.py`. In this example, we only modify the final command of the ObjectNav docker: by adding the following args to submission.sh `--model-path demo.ckpt.pth --input-type rgbd`. The default submission.sh script will pass these args to the python script. You may also replace the submission.sh.
        1. Please note that at this time, that habitat_baselines uses a slightly different config system, and the configs nodes for habitat are defined under TASK_CONFIG which is loaded at runtime from BASE_TASK_CONFIG_PATH. We manually overwrite this config using the opts args in our agent.py file.

1. Once your Dockerfile and other code is modified to your satisfaction, build it with the following command.
    ```bash
    docker build . --file Objectnav_DDPPO_baseline.Dockerfile -t objectnav_submission
    ```
1. To test locally simple run the `test_locally_objectnav_rgbd.sh` script. If the docker runs your code without errors, it should work on Eval-AI. The instructions for submitting the Docker to EvalAI are listed above.

### Hierarchical RL Starter Code
TODO


## Citing Habitat Challenge 2022
Please cite [the following paper](https://arxiv.org/abs/2006.13171) for details about the 2022 ObjectNav challenge:
```
@misc{habitatrearrangechallenge2022,
  title         =     Habitat Rearrangement Challenge 2022,
  author        =     {Andrew Szot, Karmesh Yadav, Alex Clegg, Vincent-Pierre Berges, Aaron Gokaslan, Angel Chang, Manolis Savva, Zsolt Kira, Dhruv Batra},
  howpublished  =     {\url{https://aihabitat.org/challenge/rearrange_2022}},
  year          =     {2022}
}
```

## Acknowledgments

The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team.

## References

[1] [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Manolis Savva\*, Abhishek Kadian\*, Oleksandr Maksymets\*, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.


