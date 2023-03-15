FROM fairembodied/habitat-challenge:habitat_rearrangement_2022_base_docker
ADD agents/habitat_baselines_agent.py agent.py
ADD configs/ /configs/
RUN /bin/bash -c "mkdir -p data/models"
ADD data/models/ data/models/
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/configs/tasks/rearrange_easy.local.rgbd.yaml"

RUN /bin/bash -c ". activate habitat; pip install ifcfg torchvision tensorboard"
CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && python agent.py --evaluation $AGENT_EVALUATION_TYPE --input-type depth --cfg-path configs/methods/tp_srl.yaml"]


