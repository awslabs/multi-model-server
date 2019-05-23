FROM ubuntu:16.04

LABEL maintainer="Amazon AI"

ARG MMS_VERSION=1.0.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-8-jdk-headless \
    ca-certificates \
    curl \
    wget \
    git \
    vim \
    build-essential \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.6.6

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
    ./configure && make && make install && \
    apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
    make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip

# Python wonâ€™t try to write .pyc or .pyo files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN ${PIP} --no-cache-dir install --upgrade pip setuptools

RUN ${PIP} install --no-cache-dir torch \
                                  torchvision \
                                  mxnet-model-server==$MMS_VERSION

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY mms-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY config.properties /home/model-server

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

EXPOSE 8080 8081

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
