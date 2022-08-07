<p align="center">
  <img width = "50%" src='res/img/habitat_logo_with_text_horizontal_blue.png' />
  </p>

--------------------------------------------------------------------------------

# Habitat Rearrange Challenge 2022

This repository contains the starter code for the Habitat 2022 rearrangement challenge, and training and evaluation setups. For an overview of habitat-challenge, visit [aihabitat.org/challenge/rearrange_2022](https://aihabitat.org/challenge/rearrange_2022).

## Task: Object Rearrangement

In the object rearrangement task, a Fetch robot is randomly spawned in an unknown environment and asked to rearrange 1 object from an initial to desired position – picking/placing it from receptacles (counter, sink, sofa, table), opening/closing containers (drawers, fridges) as necessary. A map of the environment is not provided and the agent must only use its sensory input to navigate and rearrange.

The Fetch robot is equipped with an egocentric 256x256 90-degree FoV RGBD camera on the robot head. 
The agent also has access to idealized base-egomotion giving the relative displacement and angle of the base since the start of the episode. 
Additionally, the robot has proprioceptive joint sensing providing access to the current robot joint angles.

For details about the agent, dataset, and evaluation, see the challenge website: [aihabitat.org/challenge/rearrange_2022](https://aihabitat.org/challenge/rearrange_2022/).

If you have any issues, open a GitHub issue on this repository.

## Participation Guidelines

Participate in the contest by registering on the in the soon to be released EvalAI page and creating a team. Participants will upload docker containers with their agents that are evaluated on an AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to ensure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Installing Habitat-Sim and Downloading data
First setup Habitat Sim in a new conda environment so you can download the datasets to evaluate your models locally.

1. Prepare your [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) env:
    ```bash
    # We require python>=3.7 and cmake>=3.10
    conda create -n habitat python=3.7 cmake=3.14.0
    conda activate habitat
    ```

1. Install Habitat-Sim using our custom Conda package for habitat challenge 2022 with: 
    ```
    conda install -y habitat-sim-rearrange-challenge-2022  withbullet  headless -c conda-forge -c aihabitat
    ```
    **On MacOS, omit the `headless` argument**.
    
    Note: If you face any issues related to the `GLIBCXX` version after conda installation, please uninstall this conda package and install the habitat-sim repository from source (more information [here](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source)). Make sure that you are using the `hab2_challenge_2022` tag and not the `stable` branch for your installation. 

1. Clone the challenge repository:

    ```bash
    git clone -b rearrangement-challenge-2022 https://github.com/facebookresearch/habitat-challenge.git
    cd habitat-challenge
    ```

1. Download the dataset with 
    ```
    python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path <path to download folder>
    ```
    If this step was successful, you should see the train, val and minival splits in the `<path to download folder>/datasets/replica_cad/rearrange/v1/{train, val, minival}` folders respectively. 

1. Now, create a symlink to the downloaded data in your habitat-challenge repository:
    ```
    mkdir -p data
    ln -s <absolute path to download folder> data
    ```



### Local Docker Evaluation
In these steps, we will evaluate a sample agent in Docker. We evaluate in Docker because EvalAI requires submitting a Docker image to run your agent on the leaderboard. **Since these steps depend on [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker), they will only run on Linux**; no Windows or MacOS.

1. Implement your own agent or try one of ours. We provide an agent in `agents/random_agent.py` that takes random actions.

    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to `RandomAgent` agent in `agents/random_agent.py`.

1. Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following instructions here: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
Note: only supports Linux; no Windows or MacOS.

1. Modify the provided Dockerfile if you need custom modifications. Let’s say your code needs `pytorch==1.9.0`, these dependencies should be pip installed inside the Docker conda environment called `habitat`. Below is an example Dockerfile with pip installing custom dependencies. 

    ```dockerfile
    FROM fairembodied/habitat-challenge:habitat_rearrangement_2022_base_docker

    # install dependencies in the habitat conda environment
    RUN /bin/bash -c ". activate habitat; pip install torch==1.9.0"

    ADD agent.py /agent.py
    ADD submission.sh /submission.sh
    ```
    Build your docker container using: 
    ```bash
    docker build . --file docker/hab2.Dockerfile  -t rearrange_submission
    ```

    Note #1: you may need `sudo` privileges to run this command.

    Note #2: Please make sure that you keep your local version of `fairembodied/habitat-challenge:habitat_rearrangement_2022_base_docker` image up to date with the image we have hosted on [dockerhub](https://hub.docker.com/r/fairembodied/habitat-challenge/tags). This can be done by pruning all cached images, using:
    ```
    docker system prune -a
    ```

1. Evaluate your docker container locally:
    ```bash
    bash ./scripts/test_local.sh --docker-name rearrange_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2022-07-14 22:03:05,811 Initializing dataset RearrangeDataset-v0
    2022-07-14 22:03:05,811 Rearrange task assets are not downloaded locally, downloading and extracting now...
    2022-07-14 22:03:05,812 Downloaded and extracted the data.
    2022-07-14 22:03:05,818 initializing sim RearrangeSim-v0
    2022-07-14 22:03:06,214 Initializing task RearrangeCompositeTask-v0
    2022-07-14 22:03:08,822 object_to_goal_distance/30: 4.302241203188896
    2022-07-14 22:03:08,822 robot_force/accum: 657186.0750969499
    2022-07-14 22:03:08,823 robot_force/instant: 657193.0133517366
    2022-07-14 22:03:08,823 force_terminate: 0.05
    2022-07-14 22:03:08,823 robot_collisions/total_collisions: 0.0
    2022-07-14 22:03:08,823 robot_collisions/robot_obj_colls: 0.0
    2022-07-14 22:03:08,823 robot_collisions/robot_scene_colls: 0.0
    2022-07-14 22:03:08,823 robot_collisions/obj_scene_colls: 0.0
    2022-07-14 22:03:08,823 ee_to_object_distance/30: 4.221194922924042
    2022-07-14 22:03:08,823 does_want_terminate: 1.0
    2022-07-14 22:03:08,823 composite_success: 0.0
    2022-07-14 22:03:08,823 composite_bad_called_terminate: 1.0
    2022-07-14 22:03:08,823 num_steps: 1.0
    2022-07-14 22:03:08,823 did_violate_hold_constraint: 0.0
    2022-07-14 22:03:08,823 move_obj_reward: -0.5103676795959473
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.**

### Online submission (COMING SOON)

Online submission on EvalAI will be announced soon!

### DD-PPO Training Starter Code
In this example, we will train and evaluate an end-to-end policy trained with DD-PPO. You will run all the subsequent steps from the `habitat` conda environment.

1. Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/) - Use the `challenge_tasks` branch in our Github repo, which can be cloned using: 
    ```
    git clone --branch challenge_tasks https://github.com/facebookresearch/habitat-lab.git
    ``` 
    Install Habitat Lab along with the included RL trainer code by first entering the `habitat-lab` directory, activating the `habitat` conda environment from step 1, and then running `pip install -r requirements.txt && python setup.py develop --all`. 

1. Follow this documentation for how to run DD-PPO in a single or multi-machine setup. See [habitat_baselines/ddppo](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines/rl/ddppo) for more information. These commands assume `habitat-lab` and `habitat-challenge` are in the same directory. Modify the paths in the arguments if your `habitat-challenge` directory is located somewhere else.

    1. To run on a single machine use the following script from `habitat-lab` directory:
        ```bash
        #/bin/bash

        export MAGNUM_LOG=quiet
        export HABITAT_SIM_LOG=quiet

        set -x
        python habitat_baselines/run.py \
            --exp-config ../habitat-challenge/configs/methods/ddppo_monolithic.yaml \
            --run-type train \
            BASE_TASK_CONFIG_PATH ../habitat-challenge/configs/tasks/rearrange.local.rgbd.yaml \
            TASK_CONFIG.DATASET.SPLIT 'train' \
            TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH ../habitat-challenge/configs/pddl/ \
            TENSORBOARD_DIR ./tb \
            CHECKPOINT_FOLDER ./checkpoints \
            LOG_FILE ./train.log
        ```
    1. To run on a cluster with SLURM using distributed training run the following script. While this is not necessary, if you have access to a cluster, it can significantly speed up training. To run on multiple machines in a SLURM cluster run the following script: change `#SBATCH --nodes $NUM_OF_MACHINES` to the number of machines and `#SBATCH --ntasks-per-node $NUM_OF_GPUS` and `$SBATCH --gres $NUM_OF_GPUS` to specify the number of GPUS to use per requested machine.
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

        export MAGNUM_LOG=quiet
        export HABITAT_SIM_LOG=quiet

        export MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

        set -x
        srun python -u -m habitat_baselines.run \
            habitat_baselines/run.py \
            --exp-config ../habitat-challenge/configs/methods/ddppo_monolithic.yaml \
            --run-type train \
            BASE_TASK_CONFIG_PATH ../habitat-challenge/configs/tasks/rearrange.local.rgbd.yaml \
            TASK_CONFIG.DATASET.DATA_PATH ../habitat-challenge/data/datasets/replica_cad/rearrange/v1/{split}/rearrange.json.gz \
            TASK_CONFIG.DATASET.SCENES_DIR ../habitat-challenge/data/replica_cad/ \
            TASK_CONFIG.DATASET.SPLIT 'train' \
            TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH ../habitat-challenge/configs/pddl/ \
            TENSORBOARD_DIR ./tb \
            CHECKPOINT_FOLDER ./checkpoints \
            LOG_FILE ./train.log
        ```

1. More instructions on how to train the DD-PPO model can be found in [habitat-lab/habitat_baselines/rl/ddppo](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines/rl/ddppo). See the corresponding README in habitat-lab for how to adjust the various hyperparameters, save locations, visual encoders and other features.

1. Evaluate on the minival dataset for the `rearrange_easy` task from the command line via. First enter the `habitat-challenge` directory. Ensure, you have the datasets installed in this directory as well. If not, run `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`.
    ```bash
    CHALLENGE_CONFIG_FILE=configs/tasks/rearrange_easy.local.rgbd.yaml python agents/habitat_baselines_agent.py --evaluation local --input-type depth --cfg-path configs/methods/ddppo_monolithic.yaml
    ```

1. We provide Dockerfiles ready to use with the DD-PPO baselines in `docker/hab2_monolithic.Dockerfile`. For the sake of completeness, we describe how you can make your own Dockerfile below. If you just want to test the baseline code, feel free to skip this bullet because  `hab2_DDPPO_baseline.Dockerfile` is ready to use.
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

    1. Finally, modify the submission.sh script to run the appropriate command to test your agents. The scaffold for this code can be found `agents/random_agent.py` and the code for policies trained with Habitat Baselines can be found in `agents/habitat_baselines_agent.py`. In this example, we only modify the final command of the docker: by adding the following args to submission.sh `--model-path demo.ckpt.pth --input-type rgbd`. The default submission.sh script will pass these args to the python script. You may also replace the submission.sh.

1. Once your Dockerfile and other code is modified to your satisfaction, build it with the following command.
    ```bash
    docker build . --file docker/hab2_monolithic.Dockerfile.Dockerfile -t rearrange_submission
    ```
1. To test locally simple run the `scripts/test_local.sh` script. If the docker runs your code without errors, it should work on Eval-AI. The instructions for submitting the Docker to EvalAI are listed above.

### Hierarchical RL Starter Code
First, you will need to train individual skill policies with RL. In this example we will approach the `rearrange_easy` task by training a Pick, Place, and Navigation policy and then plug them into a hard-coded high-level controller.
1. Follow steps 1,2,3 of [the DD-PPO section](https://github.com/facebookresearch/habitat-challenge/tree/rearrangement-challenge-2022#dd-ppo-training-starter-code) to install Habitat-Sim, install Habitat-Lab, and download the datasets.
1. Steps to train the skills from scratch:

    1. Train the Pick skill. From the Habitat Lab directory, run 
    ```bash
    python habitat_baselines/run.py \
        --exp-config habitat_baselines/config/rearrange/ddppo_pick.yaml \
        --run-type train \
        TENSORBOARD_DIR ./pick_tb/ \
        CHECKPOINT_FOLDER ./pick_checkpoints/ \
        LOG_FILE ./pick_train.log
    ```
    1. Train the Place skill. Use the exact same command as the above, but replace every instance of "pick" with "place".
    1. Train the Navigation skill. Use the exact same command as the above, but replace every instance of "pick" with "nav_to_obj".
    1. Copy the checkpoints for the different skills to the `data/models` directory in the Habitat Challenge directory. There should now be three files `data/models/[nav,pick,place].pth`.

1. Instead of training the skills, you can also use the pre-trained skills located at [this Google Drive link.](https://drive.google.com/drive/folders/1F-T5zJvz-EIzh9waDvMnuwCmkxztvaFG?usp=sharing)

1. Finally, evaluate the combined policies on the minival dataset for the `rearrange_easy` task from the command line. First enter the `habitat-challenge` directory. Ensure, you have the datasets installed in this directory as well. If not, run `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`.
    ```bash
    CHALLENGE_CONFIG_FILE=configs/tasks/rearrange_easy.local.rgbd.yaml python agents/habitat_baselines_agent.py --evaluation local --input-type depth --cfg-path configs/methods/tp_srl.yaml
    ```
    Using the pre-trained skills from the Google Drive, you should see around a `30%` success rate.


## Citing Habitat Challenge 2022
Please cite the challenge and [the following paper](https://arxiv.org/abs/2006.13171) for details about the 2022 Rearrangement challenge:
```
@misc{habitatrearrangechallenge2022,
  title         =     Habitat Rearrangement Challenge 2022,
  author        =     {Andrew Szot, Karmesh Yadav, Alex Clegg, Vincent-Pierre Berges, Aaron Gokaslan, Angel Chang, Manolis Savva, Zsolt Kira, Dhruv Batra},
  howpublished  =     {\url{https://aihabitat.org/challenge/rearrange_2022}},
  year          =     {2022}
}
```

```
@article{szot2021habitat,
  title={Habitat 2.0: Training home assistants to rearrange their habitat},
  author={Szot, Andrew and Clegg, Alexander and Undersander, Eric and Wijmans, Erik and Zhao, Yili and Turner, John and Maestre, Noah and Mukadam, Mustafa and Chaplot, Devendra Singh and Maksymets, Oleksandr and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={251--266},
  year={2021}
}
```

## Acknowledgments

The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team.

## References

[1] [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Manolis Savva\*, Abhishek Kadian\*, Oleksandr Maksymets\*, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.


