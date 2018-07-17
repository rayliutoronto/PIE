FROM tensorflow/tensorflow:1.9.0-gpu-py3

ADD . /src
WORKDIR /src

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_lg && python -m spacy link en_core_web_lg en

RUN mkdir data/word_vectors && \
    apt-get update && apt-get install -y wget unzip && \
    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip && \
    unzip glove.6B.zip && \
    cp glove.6B.50d.txt data/word_vectors && rm glove.*.txt
