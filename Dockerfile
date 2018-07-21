FROM tensorflow/tensorflow:1.9.0-gpu-py3

#nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04


ADD . /src
WORKDIR /src

RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install -r requirements.txt


RUN python -m spacy download en_core_web_lg && python -m spacy link en_core_web_lg en

RUN mkdir data/word_vectors && \
    apt-get install -y wget unzip && \
    wget --quiet http://nlp.stanford.edu/data/wordvecs/glove.6B.zip && \
    unzip glove.6B.zip && mv glove.6B.50d.txt data/word_vectors && \
    rm glove.* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/src
