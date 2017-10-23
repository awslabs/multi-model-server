#!/usr/bin/env bash

apt-get install -y openjdk-7-jdk
apt-get install -y curl
apt-get install -y nginx

pip install gunicorn
pip install gevent
pip install deep-model-server

