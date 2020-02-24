FROM fairembodied/habitat-challenge:2020

# install dependencies in the habitat conda environment
RUN /bin/bash -c ". activate habitat"

#ADD baselines /baselines
ADD agent.py /agent.py
ADD submission.sh /submission.sh
