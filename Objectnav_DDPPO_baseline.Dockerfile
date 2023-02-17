FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

RUN /bin/bash -c ". activate habitat; pip install git+https://github.com/openai/CLIP.git@40f5484c1c74edd83cb9cf687c6ab92b28d8b656"
ADD ddppo_agents.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ADD configs/ configs/
ADD fp_exp_ver_rgb_clip_hm3d_finetuned_val_best_39.pth demo.ckpt.pth
ADD clip_policy.py clip_policy.py
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path demo.ckpt.pth --input-type rgb"]
