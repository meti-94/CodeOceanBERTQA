# hash:sha256:0fad6d02a3cf30127afa93210140d042a3c9d4179d866ae3301d04d98b729aad
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip=20.0.2-5ubuntu1.8 \
        python3-setuptools=45.2.0-1ubuntu0.1 \
        python3-wheel=0.34.2-1ubuntu0.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir jupyter==1.0.0 
RUN pip3 install -U --no-cache-dir nltk
RUN pip3 install -U --no-cache-dir torch 
RUN pip3 install -U --no-cache-dir torchvision 
RUN pip3 install -U --no-cache-dir torchaudio
RUN pip3 install -U --no-cache-dir torchtext==0.2.3  
RUN pip3 install -U --no-cache-dir pytorch-crf==0.7.2  
RUN pip3 install -U --no-cache-dir fuzzywuzzy  
