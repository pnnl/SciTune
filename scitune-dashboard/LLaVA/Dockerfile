FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Mount volume 
ARG FILE_PATH="/home/ubuntu/scitune_data/"
ENV FILE_PATH=${FILE_PATH}
VOLUME ${FILE_PATH}:${FILE_PATH}
RUN chmod 777 -R ${FILE_PATH}



RUN apt update && apt install -y \
    wget

# Miniconda setup
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH="/root/miniconda3/bin:${PATH}"

# Copy Dashboard files
copy . /LLaVA


























# RUN conda init 

# ### LLAVA setup
# # Create a directory
# RUN mkdir /llava_test

# # Set working directory
# WORKDIR /llava_test

# RUN apt update && apt install -y \
#     git
# # Initialize Git repository
# RUN git init


# # Clone the repository
# RUN git clone https://github.com/haotian-liu/LLaVA.git

# # Set working directory to LLaVA
# WORKDIR /llava_test/LLaVA

# # Create Conda environment
# RUN conda create -n llava python=3.10 -y

# # Activate Conda environment
# SHELL ["conda", "run", "-n", "llava", "/bin/bash", "-c"]

# # Upgrade pip
# RUN pip install --upgrade pip

# # Install LLaVA package
# RUN pip install -e .



# # Install flash-attn
# RUN pip install flash-attn --no-build-isolation



# # Expose ports
# EXPOSE 10000
# EXPOSE 40000
# EXPOSE 7860

# # Copy the shell script
# COPY start.sh /llava_test/LLaVA/start.sh

# # Set execute permissions for the script
# RUN chmod +x /llava_test/LLaVA/start.sh

# # Command to run the shell script
# CMD ["/llava_test/LLaVA/start.sh"]

# ENTRYPOINT ["conda", "run", "-n", "myenv", "bash", "/llava_test/LLaVA/start.sh"]

    
# # Nvidia setup
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt update && \
#     apt install software-properties-common -y && \
#     apt-get update -y && \
#     apt-get upgrade -y && \
#     add-apt-repository ppa:graphics-drivers && \
#     apt install nvidia-driver-440 -y


# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
