FROM fairembodied/habitat-challenge:habitat_rearrangement_2022_base_docker
ADD agents/habitat_baselines_agent.py agent.py
ADD configs/ /configs/
ADD scripts/submission.sh submission.sh
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/configs/tasks/rearrange.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh"]
