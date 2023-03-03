FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

ADD agents/agent.py agent.py
ADD configs/ /configs/
ADD submission.sh /submission.sh

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml"

CMD [ \
    "/bin/bash", \
    "-c", \
    "source activate habitat && \
    export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && \
    export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && \
    python agent.py --evaluation $AGENT_EVALUATION_TYPE" \
]
