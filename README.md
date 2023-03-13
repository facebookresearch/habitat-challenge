<p align="center">
  <img width = "50%" src='res/img/habitat_logo_with_text_horizontal_blue.png' />
  </p>

--------------------------------------------------------------------------------

# Habitat Navigation Challenge 2023

This repository contains the starter code for the 2023 Habitat [1] challenge, details of the tasks, and training and evaluation setups. For an overview of habitat-challenge, visit [aihabitat.org/challenge](https://aihabitat.org/challenge/).

If you are looking for our 2022/2021/2020/2019 starter code, it’s available in the [`challenge-YEAR branch`](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2022).

This year, we are hosting a challenges on the ObjectNav and ImageNav embodied navigation task. 

Task #1: ObjectNav focuses on egocentric object/scene recognition and a commonsense understanding of object semantics (where is a bed typically located in a house?).

Task #2: ImageNav focuses on visual reasoning and embodied instance disambiguation (is the particular chair I observe the same one depicted by the goal image?).

### New in 2023
- We are instantiating ObjectNav on a new version of the `HM3D-Semantics` dataset called [HM3D-Semantics v0.2](https://aihabitat.org/datasets/hm3d-semantics/).
- We are announcing the ImageNav track, also on the [HM3D-Semantics v0.2](https://aihabitat.org/datasets/hm3d-semantics/) scene dataset.
- We are introducing several changes in the agent config for easier sim-to-real transfer. We are using the [HelloRobot Stretch](https://hello-robot.com/stretch-2) robot configuration with support of continuous action space and updating the dataset such that all episodes can be navigated without traversing between floors.


## Task: ObjectNav

In ObjectNav, an agent is initialized at a random starting position and orientation in an unseen environment and asked to find an instance of an object category (*‘find a chair’*) by navigating to  it. A map of the environment is not provided and the agent must only use its sensory input to navigate. 

The agent is modeled after the Hello Stretch robot and equipped with an RGB-D camera and a (noiseless) GPS+Compass sensor. GPS+Compass sensor provides the agent’s current location and orientation information relative to the start of the episode. 

### Dataset
The 2023 ObjectNav challenge uses 216 scenes from the [HM3D-Semantics v0.2](https://aihabitat.org/datasets/hm3d-semantics/) [2] dataset with train/val/test splits on 145/36/35. Following Chaplot et al. [3], we use 6 object goal categories: chair, couch, potted plant, bed, toilet and tv. All episodes can be navigated without traversing between floors.

## Task: ImageNav

In ImageNav, an agent is initialized at a random start pose in an unseen environment and given an RGB goal image. We adopt the [Instance ImageNav](https://arxiv.org/abs/2211.15876) [4] task definition where the goal image depicts a particular object instance and the agent is asked to navigate to that object. 

The goal camera is disentangled from the agent's camera; sampled parameters such as height, look-at-angle, and field-of-view reflect the realistic use case of a user-supplied goal image.

Similar to ObjectNav, the agent is modeled after the Hello Stretch robot and equipped with an RGB-D camera and a (noiseless) GPS+Compass sensor.

### Dataset
The 2023 ImageNav challenge uses 216 scenes from the [HM3D-Semantics v0.2](https://aihabitat.org/datasets/hm3d-semantics/)[2] dataset with train/val/test splits on 145/36/35. Following Krantz et al. [4], we sample goal images depicting object instances belonging to the same 6 goal categories used in the ObjectNav challenge: chair, couch, potted plant, bed, toilet, and tv. All episodes can be navigated without traversing between floors.

## Evaluation
Similar to 2022 Habitat Challenge, we measure performance along the same two axes as specified by Anderson et al.[4]:
- **Success**: Did the agent navigate to an instance of the goal object? (Notice: *any* instance, regardless of distance from starting location.)

    Concretely, an episode is deemed successful if on calling the STOP action, the agent is within 1.0m Euclidean distance from any instance of the target object category AND the object *can be viewed by an oracle* from that stopping position by turning the agent or looking up/down. Notice: we do NOT require the agent to be actually viewing the object at the stopping location, simply that such oracle-visibility is possible without moving. Why? Because we want participants to focus on *navigation*, not object framing. In Embodied AI’s larger goal, the agent is navigating to an object instance to interact with it (say point at or manipulate an object). Oracle-visibility is our proxy for *‘the agent is close enough to interact with the object’*.

- **SPL**: How efficient was the agent’s path compared to an optimal path? (Notice: for ObjectNav, optimal path = shortest path from the agent’s starting position to the *closest* instance of the target object category.)

After calling the STOP action, the agent is evaluated using the ‘Success weighted by Path Length’ (SPL) metric [4].

<p align="center">
  <img src='res/img/spl.png' />
</p>

ObjectNav-SPL is defined analogous to PointNav-SPL. The only key difference is that the shortest path is computed to the object instance closest to the agent start location. Thus, if an agent spawns very close to *‘chair1’* but stops at a distant *‘chair2’*, it will achieve 100% success (because it found a *‘chair’*) but a fairly low SPL (because the agent path is much longer compared to the oracle path). ImageNav-SPL is similar to ObjectNav-SPL except that there is exactly one correct object instance (shown in the goal image).

We reserve the right to use additional metrics to choose winners in case of statistically insignificant SPL differences.

## Participation Guidelines

Coming Soon!
<!-- Participate in the contest by registering on the [EvalAI challenge page](https://eval.ai/web/challenges/challenge-page/1615/overview) and creating a team. Participants will upload docker containers with their agents that are evaluated on an AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to ensure it is working. Instructions for training, local evaluation, and online submission are provided below.

For your convenience, please check our [Habitat Challenge video tutorial](https://youtu.be/V7PXttmJ8EE?list=PLGywud_-HlCORC0c4uj97oppQrGiB6JNy) and [Colab step-by-step tutorial from previous year](https://colab.research.google.com/gist/mathfac/8c9b97d7afef36e377f17d587c903ede). -->

### Local Evaluation

1. Clone the challenge repository:

    ```bash
    git clone https://github.com/facebookresearch/habitat-challenge.git
    cd habitat-challenge
    ```

1. Implement your own agent or try one of ours. We provide an agent in `agents/agent.py` that takes random actions:
    ```python
    import habitat
    from omegaconf import DictConfig

    class RandomAgent(habitat.Agent):
        def __init__(self, task_config: DictConfig):
            self._task_config = task_config

        def reset(self):
            pass

        def act(self, observations):
            return {
                'action': ("velocity_control", "velocity_stop"),
                'action_args': {
                    "angular_velocity": np.random.rand(1),
                    "linear_velocity": np.random.rand(1),
                    "velocity_stop": np.random.rand(1),
                }
            }


    def main():
        agent = RandomAgent(task_config=config)
        challenge = habitat.Challenge()
        challenge.submit(agent)
    ```


1. Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following instructions here: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
Note: only supports Linux; no Windows or MacOS.

1. Modify the provided Dockerfile (`docker/ObjectNav_random_baseline.Dockerfile`) if you need custom modifications. Let’s say your code needs `pytorch`, these dependencies should be pip installed inside a conda environment called `habitat` that is shipped with our habitat-challenge docker, as shown below:

    ```dockerfile
    FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

    # install dependencies in the habitat conda environment
    RUN /bin/bash -c ". activate habitat; pip install torch"

    ADD agents/agent.py /agent.py
    ADD submission.sh /submission.sh
    ```
    Build your docker container using: `docker build . --file docker/ObjectNav_random_baseline.Dockerfile -t objectnav_submission`. (Note: you may need `sudo` priviliges to run this command.)

    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to `RandomAgent` agent in `agent.py`


1. Scene Dataset: Download Habitat-Matterport3D Dataset scenes used for Habitat Challenge [here](https://matterport.com/partners/facebook). Place this data in: `habitat-challenge/habitat-challenge-data/data/scene_datasets/hm3d_v0.2`

    **Using Symlinks:**  If you used symlinks (i.e. `ln -s`) to link to an existing download of HM3D, there is an additional step. First, make sure there is only one level of symlink (instead of a symlink to a symlink link to a .... symlink) with
      ```bash
      ln -f -s $(realpath habitat-challenge-data/data/scene_datasets/hm3d_v0.2) \
          habitat-challenge-data/data/scene_datasets/hm3d_v0.2
      ```

    Then modify the docker command in `scripts/test_local_objectnav.sh` file to mount the linked to location by adding `-v $(realpath habitat-challenge-data/data/scene_datasets/hm3d_v0.2):/habitat-challenge-data/data/scene_datasets/hm3d_v0.2`. The modified docker command would be
     ```bash
    docker run \
          -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
          -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d_v0.2):/habitat-challenge-data/data/scene_datasets/hm3d_v0.2 \
          --runtime=nvidia \
          -e "AGENT_EVALUATION_TYPE=local" \
          -e "TRACK_CONFIG_FILE=/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml" \
          ${DOCKER_NAME}
    ```

1. Evaluate your docker container locally:
    ```bash
    ./scripts/test_local_objectnav.sh --docker-name objectnav_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2023-03-01 16:35:02,244 distance_to_goal: 6.446822468439738
    2023-03-01 16:35:02,244 success: 0.0
    2023-03-01 16:35:02,244 spl: 0.0
    2023-03-01 16:35:02,244 soft_spl: 0.0014486297806195665
    2023-03-01 16:35:02,244 num_steps: 1.0
    2023-03-01 16:35:02,244 collisions/count: 0.0
    2023-03-01 16:35:02,244 collisions/is_collision: 0.0
    2023-03-01 16:35:02,244 distance_to_goal_reward: 0.0009365876515706381
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.**

### Online submission

Follow instructions in the `submit` tab of the EvalAI challenge page (coming soon) to submit your docker image. Note that you will need a version of EvalAI `>= 1.2.3`. Pasting those instructions here for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.3.5"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push objectnav_submission:latest --phase <phase-name>
```

The challenge consists of the following phases:

1. **Minival phase**: This split is same as the one used in `./scripts/test_local_objectnav.sh`. The purpose of this phase/split is sanity checking -- to confirm that our remote evaluation reports the same result as the one you’re seeing locally. Each team is allowed maximum of 100 submissions per day for this phase, but please use them judiciously. We will block and disqualify teams that spam our servers.
1. **Test Standard phase**: The purpose of this phase/split is to serve as the public leaderboard establishing the state of the art; this is what should be used to report results in papers. Each team is allowed maximum of 10 submissions per day for this phase, but again, please use them judiciously. Don’t overfit to the test set.
1. **Test Challenge phase**: This phase/split will be used to decide challenge winners. Each team is allowed a total of 5 submissions until the end of challenge submission phase. The highest performing of these 5 will be automatically chosen. Results on this split will not be made public until the announcement of final results at the [Embodied AI workshop at CVPR](https://embodied-ai.org/).

Note: Your agent will be evaluated on 1000 episodes and will have a total available time of 48 hours to finish. Your submissions will be evaluated on AWS EC2 p2.xlarge instance which has a Tesla K80 GPU (12 GB Memory), 4 CPU cores, and 61 GB RAM. If you need more time/resources for evaluation of your submission please get in touch. If you face any issues or have questions you can ask them by opening an issue on this repository.

### ObjectNav Baselines and DD-PPO Training Starter Code
We have added a config in `configs/ddppo_objectnav_v2_hm3d_stretch.yaml` that includes a baseline using DD-PPO from Habitat-Lab.

1. Install the [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/) packages. You can install Habitat-Sim using our custom Conda package for habitat challenge 2023 with: ```conda install -c aihabitat habitat-sim-challenge-2023```. For Habitat-Lab, we have created the `habitat-challenge-2023` tag in our Github repo, which can be cloned using: ```git clone --branch challenge-2023 https://github.com/facebookresearch/habitat-lab.git```. Please ensure that both habitat-lab and habitat-baselines packages are installed using ```pip install -e habitat-lab``` and ```pip install -e habitat-baselines```. You will find further information for installation in the Github repositories. 

1. Download the HM3D scene dataset following the instructions [here](https://matterport.com/partners/facebook). After downloading extract the dataset to folder `habitat-lab/data/scene_datasets/hm3d_v0.2/` folder (this folder should contain the `.glb` files from HM3D). Note that the `habitat-lab` folder is the [habitat-lab](https://github.com/facebookresearch/habitat-lab/) repository folder. You could also just symlink to the path of the HM3D scenes downloaded in step-4 of local-evaluation under the `habitat-challenge/habitat-challenge-data/data/scene_datasets` folder. This can be done using `ln -s /path/to/habitat-challenge-data/data/scene_datasets /path/to/habitat-lab/data/scene_datasets/` (if on OSX or Linux).

1. **Objectnav**: Download the episodes dataset for HM3D ObjectNav from [link](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip) and place it in the folder `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/hm3d`. If placed correctly, you should have the train and val splits at `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/hm3d/v2/train/` and `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/hm3d/v2/val/` respectively.

1. An example on how to train DD-PPO model can be found in [habitat-lab/habitat-baselines/habitat_baselines/rl/ddppo](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines/rl/ddppo). See the corresponding README in habitat-lab for how to adjust the various hyperparameters, save locations, visual encoders and other features.

    1. To run on a single machine use the script [single_node.sh](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/ddppo/single_node.sh) from the `habitat-lab` directory:
        ```bash
        #/bin/bash

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet

        set -x

        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node 1 \
            habitat_baselines/run.py \
            --config-name=objectnav/ddppo_objectnav_v2_hm3d_stretch.yaml
        ```
    1. There is also an example script named [multi_node_slurm.sh](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/ddppo/multi_node_slurm.sh) for running the code in distributed mode on a cluster with SLURM. While this is not necessary, if you have access to a cluster, it can significantly speed up training. To run on multiple machines in a SLURM cluster run the following script: change ```#SBATCH --nodes $NUM_OF_MACHINES``` to the number of machines and ```#SBATCH --ntasks-per-node $NUM_OF_GPUS``` and ```$SBATCH --gpus $NUM_OF_GPUS``` to specify the number of GPUS to use per requested machine.
        ```bash
        #!/bin/bash
        #SBATCH --job-name=ddppo
        #SBATCH --output=logs.ddppo.out
        #SBATCH --error=logs.ddppo.err
        #SBATCH --gpus 1
        #SBATCH --nodes 1
        #SBATCH --cpus-per-task 10
        #SBATCH --ntasks-per-node 1
        #SBATCH --mem=60GB
        #SBATCH --time=72:00:00
        #SBATCH --signal=USR1@90
        #SBATCH --requeue
        #SBATCH --partition=dev

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet

        MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
        export MAIN_ADDR

        set -x
        srun python -u -m habitat_baselines.run \
            --config-name=configs/ddppo_objectnav_v2_hm3d_stretch.yaml
        ```

1. The checkpoint specified by ```$PATH_TO_CHECKPOINT ``` can evaluated based on the SPL and other measurements by running the following command:

    ```bash
    python -u -m habitat_baselines.run \
        --config-name=configs/ddppo_objectnav_v2_hm3d_stretch.yaml \
        habitat_baselines.evaluate=True \
        habitat_baselines.eval_ckpt_path_dir=$PATH_TO_CHECKPOINT \
        habitat.dataset.data_path.split=val
    ```
    The weights used for our DD-PPO Objectnav baseline for the Habitat-2023 challenge can be downloaded with the following command:
    ```bash
    wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v2/ddppo_objectnav_habitat2023_challenge_baseline_v1.pth
    ```

1. To submit your entry via EvalAI, you will need to build a docker file. We provide Dockerfiles ready to use with the DD-PPO baselines in `docker/ObjectNav_ddppo_baseline.Dockerfile`. For the sake of completeness, we describe how you can make your own Dockerfile below. If you just want to test the baseline code, feel free to skip this bullet because  ```ObjectNav_ddppo_baseline.Dockerfile``` is ready to use.
    1. You may want to modify the `ObjectNav_ddppo_baseline.Dockerfile` to include PyTorch or other libraries. To install pytorch, ifcfg and tensorboard, add the following command to the Docker file:
        ```dockerfile
        RUN /bin/bash -c ". activate habitat; pip install ifcfg torch tensorboard"
        ```
    1. You change which ```agent.py``` and which ``submission.sh`` script is used in the Docker, modify the following lines and replace the first agent.py or submission.sh with your new files.:
        ```dockerfile
        ADD agents/agent.py agent.py
        ADD submission.sh submission.sh
        ```
    1. Do not forget to add any other files you may need in the Docker, for example, we add the ```demo.ckpt.pth``` file which is the saved weights from the DD-PPO example code.

    1. Finally, modify the submission.sh script to run the appropriate command to test your agents. The scaffold for this code can be found ```agent.py``` and the DD-PPO specific agent can be found in ```habitat_baselines_agents.py```. In this example, we only modify the final command of the ObjectNav docker: by adding the following args to submission.sh ```--model-path demo.ckpt.pth --input-type rgbd```. The default submission.sh script will pass these args to the python script. You may also replace the submission.sh.

1. Once your Dockerfile and other code is modified to your satisfaction, build it with the following command.
    ```bash
    docker build . --file docker/ObjectNav_ddppo_baseline.Dockerfile -t objectnav_submission
    ```
1. To test locally simple run the ```scripts/test_local_objectnav.sh``` script. If the docker runs your code without errors, it should work on Eval-AI. The instructions for submitting the Docker to EvalAI are listed above.
1. Happy hacking!

## Citing Habitat Challenge 2023
Please cite the following bibtex when referring to the 2023 Navigation challenge:
```
@misc{habitatchallenge2023,
  title         =     Habitat Challenge 2023,
  author        =     {Karmesh Yadav and Jacob Krantz and Ram Ramrakhya and Santhosh Kumar Ramakrishnan and Jimmy Yang and Austin Wang and John Turner and Aaron Gokaslan and Oleksandr Maksymets and Angel X Chang and Manolis Savva and Alexander Clegg and Devendra Singh Chaplot and Dhruv Batra},
  howpublished  =     {\url{https://aihabitat.org/challenge/2023/}},
  year          =     {2023}
}
```

## Acknowledgments

The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team. We also thank the team behind [Habitat-Matterport3D](https://aihabitat.org/datasets/hm3d/) and [HM3D-Semantics](https://aihabitat.org/datasets/hm3d-semantics/) datasets.

## References

[1] [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Manolis Savva\*, Abhishek Kadian\*, Oleksandr Maksymets\*, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[2] [Habitat-Matterport 3D Semantics Dataset (HM3DSem)](https://arxiv.org/abs/2210.05633). Karmesh Yadav\*, Ram Ramrakhya\*, Santhosh Kumar Ramakrishnan\*, Theo Gervet, John Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, Alexander William Clegg\^, Devendra Singh Chaplot\^. arXiv:2210.05633, 2022.

[3] [Object Goal Navigation using Goal-Oriented Semantic Exploration](https://arxiv.org/abs/2007.00643) Devendra Singh Chaplot, Dhiraj Gandhi, Abhinav Gupta, Ruslan Salakhutdinov. NeurIPS, 2020.

[4] [Instance-Specific Image Goal Navigation: Training Embodied Agents to Find Object Instances](https://arxiv.org/abs/2211.15876). Jacob Krantz, Stefan Lee, Jitendra Malik, Dhruv Batra, Devendra Singh Chaplot. arxiv:2211.15876, 2022.

[5] [On evaluation of embodied navigation agents](https://arxiv.org/abs/1807.06757). Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, Amir R. Zamir. arXiv:1807.06757, 2018.
