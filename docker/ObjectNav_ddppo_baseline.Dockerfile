FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

ADD agents/habitat_baselines_agent.py agent.py
ADD configs/ /configs/
ADD submission.sh /submission.sh
ADD data/models/objectnav_baseline_habitat_navigation_challenge_2023.pth demo.ckpt.pth

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml"

RUN /bin/bash -c ". activate habitat; pip install ifcfg torchvision tensorboard"

CMD [ \
    "/bin/bash", \
    "-c", \
    "source activate habitat && \
    export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && \
    export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && \
    bash submission.sh \
    --model-path demo.ckpt.pth \
    --input-type rgbd \
    --task-config configs/ddppo_objectnav_v2_hm3d_stretch.yaml"
]