FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

ADD agents/agent.py agent.py
ADD agents/config.py config.py
ADD configs/ /configs/
ADD scripts/submission.sh /submission.sh

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/configs/benchmark/nav/imagenav/imagenav_hm3d_v3_challenge.yaml"

CMD [ \
    "/bin/bash", \
    "-c", \
    "source activate habitat && \
    export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && \
    export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && \
    bash submission.sh \
    --action_space velocity_controller" \
]
