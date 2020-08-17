FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

RUN /bin/bash -c ". activate habitat; pip install ifcfg tensorboard && pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html"

# Update habitat_baselines to desired commit
RUN /bin/bash -c ". activate habitat; git clone http://github.com/facebookresearch/habitat-lab.git habitat-lab2 && (cd habitat-lab2 && git checkout 959bd45431edd8024832a877bdc8218015d97a7e) && cp -r habitat-lab2/habitat_baselines habitat-api/."

ADD ddppo_agents.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_pointnav2020.local.rgbd.yaml /challenge_pointnav2020.local.rgbd.yaml
ADD configs/challenge_pointnav2020.local.rgbd_test_scene.yaml /challenge_pointnav2020.local.rgbd_test_scene.yaml
ADD configs/ configs/
ADD ddppo_pointnav_habitat2020_challenge_baseline_v1.pth demo.ckpt.pth
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_pointnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path demo.ckpt.pth --input-type rgbd"]
