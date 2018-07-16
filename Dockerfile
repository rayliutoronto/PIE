FROM python:3.6.6-alpine

WORKDIR .

RUN apt-get install -y wget unzip

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_lg
RUN python -m spacy link en_core_web_lg en

RUN mkdir /tmp/cached
RUN if [[ ! -f /tmp/cached/glove.6B.zip ]]; \
    then \
    wget --quiet -P /tmp/cached/ http://nlp.stanford.edu/data/wordvecs/glove.6B.zip; \
    unzip /tmp/cached/glove.6B.zip -d /tmp/cached; \
    fi
