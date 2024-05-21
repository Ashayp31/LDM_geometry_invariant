FROM nvcr.io/nvidia/pytorch:22.11-py3

# The commented lines are meant for internal use by KCL BMEIS staff
#ARG USER_ID
#ARG GROUP_ID
#ARG USER

ENV TZ=Europe/London

#RUN addgroup --gid $GROUP_ID $USER
#RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y sudo
RUN pip3 install --upgrade pip
RUN apt-get install -y ffmpeg

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
# This is required for cases where the sqsh file is run without internet connection
RUN export TORCH_HOME=/workspace/.cache

#USER $USER

#RUN source /home/$USER/.bashrc
#RUN source /home/$USER/.profile

ENTRYPOINT ["/bin/bash"]

