#FROM ubuntu:16.04
FROM tensorflow/tensorflow:1.9.0-py3

LABEL maintainer="Ray Liu <ray.liu@toronto.ca>"

ADD . /PIE
WORKDIR /PIE

RUN apt-get update && apt-get install -y curl && \
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y python3 python3-pip tensorflow-model-server && \
    pip3 install -r requirements.txt && \
    python -m spacy download en_core_web_lg && python -m spacy link en_core_web_lg en && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/PIE

RUN chmod +x StartServing.sh
EXPOSE 19999
#CMD ["/PIE/StartServing.sh"]
