FROM tensorflow/tensorflow:1.9.0-gpu-py3

#nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04


ADD . /src
WORKDIR /src

RUN apt-get update && apt-get install -y python3-pip wget unzip tensorflow-model-server && \
    pip3 install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m spacy download en_core_web_lg && python -m spacy link en_core_web_lg en

RUN mkdir data/word_vectors && \
    wget --quiet http://nlp.stanford.edu/data/wordvecs/glove.6B.zip && \
    unzip glove.6B.zip && mv glove.6B.50d.txt data/word_vectors && \
    rm glove.*

ENV PYTHONPATH=/src

RUN cd PIE && python3 data.py && python3 model.py
RUN tensorflow_model_server --port=9000 --model_name=pie --model_base_path=/PIE/output/SavedModels &
EXPOSE 9000
