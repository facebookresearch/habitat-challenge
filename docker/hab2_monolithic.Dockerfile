FROM fairembodied/habitat-challenge:habitat_rearrangement_2022_base_docker
ADD agents/habitat_baselines_agent.py agent.py
ADD configs/ /configs/
ADD data/models/rearrange_easy.pth demo.ckpt.pth
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/configs/tasks/rearrange.local.rgbd.yaml"

RUN /bin/bash -c ". activate habitat; pip install ifcfg torchvision tensorboard"
CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && python agent.py --evaluation $AGENT_EVALUATION_TYPE --model-path demo.ckpt.pth --input-type depth --cfg-path configs/methods/ddppo_monolithic.yaml"]


