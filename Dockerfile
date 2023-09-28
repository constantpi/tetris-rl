#FROM ubuntu:22.04
FROM nvidia/cuda:12.0.0-base-ubuntu22.04
# FROM tensorflow/tensorflow:nightly-gpu

# 必要そうなものをinstall
RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \ 
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev

#任意バージョンのpython install
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz \
&& tar -xf Python-3.11.4.tgz \
&& cd Python-3.11.4 \
&& ./configure --enable-optimizations\
&& make \
&& make install

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    	vim \
        inetutils-ping \
        iproute2 \
        mesa-utils \
        nano \
        net-tools \
        openssh-client \
        zip \
        unzip \
        ca-certificates
#サイズ削減のため不要なものは削除
RUN apt-get autoremove -y

RUN echo "source /root/docker_scripts/initialize-bash-shell.sh" >> /root/.bashrc

#必要なpythonパッケージをpipでインストール
RUN pip3 install --upgrade pip
RUN pip3 install tqdm matplotlib
RUN pip3 install torch tensorboardX

# Overwrite the entry point of the parent image.
ENTRYPOINT []

# Enter the container with a Bash shell.
CMD ["/bin/bash"]
