FROM brunneis/python:3.7.7-ubuntu-20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y python3-pip libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 sudo
RUN apt-get update && apt-get install -y git
RUN apt-get remove swig
RUN apt-get install swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig

COPY --chown=root . /home/user/app/

RUN mkdir /mpe

RUN git clone https://github.com/semitable/multiagent-particle-envs.git /mpe

RUN pip install -e ./mpe/

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN cd /home/user/app/ && pip3 install -r requirements.txt

WORKDIR /home/user/app

ENTRYPOINT ["python3","./src/main.py"]