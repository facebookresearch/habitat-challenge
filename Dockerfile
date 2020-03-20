FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

ENV AGENT_EVALUATION_TYPE remote
ADD agent.py agent.py
ADD submission.sh submission.sh
ADD habitat-challenge-data/challenge_pointnav2020.local.rgbd.yaml /challenge_pointnav2020.local.rgbd.yaml

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=/challenge_pointnav2020.local.rgbd.yaml && bash submission.sh"]

