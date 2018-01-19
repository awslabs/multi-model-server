#!/usr/bin/env bash

apt-get update
apt-get install -y build-essential libatlas-base-dev libopencv-dev graphviz
apt-get install -y python-setuptools python-pip
apt-get install -y openjdk-7-jdk
apt-get install -y nginx

apt-get install -y software-properties-common
add-apt-repository -y ppa:certbot/certbot
apt-get update
apt-get install -y python-certbot-nginx
apt-get install protobuf-compiler libprotoc-dev

pip install --upgrade pip
pip install gunicorn
pip install gevent
pip install mxnet-model-server
