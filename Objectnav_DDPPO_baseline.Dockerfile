FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

ADD ddppo_agents.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ADD configs/ configs/
ADD ddppo_objectnav_habitat2022_challenge_baseline_v1.pth demo.ckpt.pth
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path demo.ckpt.pth --input-type rgbd"]
