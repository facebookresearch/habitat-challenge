FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker
ADD agents/habitat_baselines_agent.py agent.py
ADD configs/ /configs/
ADD data/models/objectnav_baseline_habitat_navigation_challenge_2023.pth demo.ckpt.pth
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/configs/tasks/objectnav.local.rgbd.yaml"

RUN /bin/bash -c ". activate habitat; pip install ifcfg torchvision tensorboard"

CMD [ \
    "/bin/bash", \
    "-c", \
    "source activate habitat && \
    export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && \
    export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && \
    python agent.py --evaluation $AGENT_EVALUATION_TYPE \
    --model-path demo.ckpt.pth \
    --input-type rgbd \
    --cfg-path configs/methods/ddppo_objectnav.yaml"
]