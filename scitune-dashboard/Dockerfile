FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
RUN apt update && apt install -y \
    wget
RUN apt update && apt install -y \
    screen
RUN apt update && apt install -y \
    vim
RUN apt update && apt install -y \
    curl

    
# Miniconda setup
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH="/root/miniconda3/bin:${PATH}"


WORKDIR "/opt/scitune"
RUN mkdir "data"
RUN chmod 777 -R "/opt/scitune"

COPY . scitune-dashboard
WORKDIR "/opt/scitune/scitune-dashboard/LLaVA"

RUN conda create -n llava python=3.10 -y
RUN conda run -n llava --no-capture-output pip install --upgrade pip  
RUN conda run -n llava --no-capture-output pip install -e .



