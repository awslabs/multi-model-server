FROM awsdeeplearningteam/mms_cpu_base:latest

RUN pip install mxnet-model-server \
    && pip uninstall --yes mxnet \
    && pip install mxnet-mkl \
    && mkdir /mxnet_model_server

COPY mms_app_cpu.conf wsgi.py setup_mms.py mxnet-model-server /mxnet_model_server/

ENV PATH="/mxnet_model_server:${PATH}" MXNET_MODEL_SERVER_GPU_IMAGE=0

LABEL maintainer="dantu@amazon.com, esterhaz@amazon.com, hag@amazon.com"
