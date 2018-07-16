FROM tensorflow/tensorflow:1.8.0

ADD . /src
WORKDIR /src

RUN apt-get update && apt-get install -y wget unzip

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_lg
RUN python -m spacy link en_core_web_lg en

RUN mkdir /tmp/cached
RUN if [[ ! -f /tmp/cached/glove.6B.zip ]]; \
    then \
    wget --quiet -P /tmp/cached/ http://nlp.stanford.edu/data/wordvecs/glove.6B.zip; \
    unzip /tmp/cached/glove.6B.zip -d /tmp/cached; \
    fi
