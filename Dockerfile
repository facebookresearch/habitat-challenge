FROM fairembodied/habitat-challenge:latest

# install dependencies in the habitat conda environment
RUN /bin/bash -c ". activate habitat; pip install torch"

# install habitat-api
RUN git clone https://github.com/facebookresearch/habitat-api.git
RUN /bin/bash -c ". activate habitat; pip uninstall -y habitat; cd habitat-api; git checkout 0985c6ffd17557150488d238d79574c60612faa9; pip install ."

ADD baselines /baselines
ADD agent.py /agent.py
ADD submission.sh /submission.sh
