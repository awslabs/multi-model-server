FROM awsdeeplearningteam/mms_gpu_base:latest

RUN pip install mxnet-model-server \
    && pip uninstall --yes mxnet \
    && pip install --no-cache-dir mxnet-cu90mkl \
    && mkdir /mxnet_model_server

COPY mms_app_gpu.conf wsgi.py setup_mms.py mxnet-model-server /mxnet_model_server/

ENV PATH="/mxnet_model_server:${PATH}" MXNET_MODEL_SERVER_GPU_IMAGE=1 gpu_id=0

LABEL maintainer="dantu@amazon.com, esterhaz@amazon.com, hag@amazon.com"
