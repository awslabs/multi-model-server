# Advanced Settings

## Contents of this Document
* [GPU Inference](advanced_settings.md#gpu-inference)
* [Reference Commands](advanced_settings.md#reference-commands)
* [Docker Details](advanced_settings.md#docker-details)
* [Description of Config File Settings](advanced_settings.md#description-of-config-file-settings)
* [Configuring SSL](advanced_settings.md#configuring-ssl)


## Other Relevant Documents
* [Quickstart](README.md#quickstart)
* [Configuring MMS with Docker](README.md#configuring-mms-with-docker)



## GPU Inference

**Step 1: Install nvidia-docker.**

`nvidia-docker` is NVIDIA's customized version of Docker that makes accessing your host's GPU resources from Docker a seamless experience. All of your regular Docker commands work the same way.

Follow the [instructions for installing nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart). Return here and follow the next step when the installation completes.

**Step 2: Download the GPU configuration template.**

A GPU configuration template is provided for your use.
Download the template a GPU config and place it in the `models` folder you just created:
* [mms_app_gpu.conf](mms_app_gpu.conf)

**Step 3: Modify the configuration template.**

Edit the file you downloaded, `mms_app_gpu.conf`. It will have the following section for `MMS Arguments`:

```
[MMS Arguments]
--models
squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
...
```

To change the model, you will update the `--models` entry. Use the following or select a different model from the [model zoo](../docs/model_zoo.md) and replace the URL.

```
resnet-18=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model
```

You may also download the model and use the shared file path, but specify the path from within the Docker container.

```
resnet-18=/models/resnet-18.model
```

Save the file.

**Step 4: Run MMS with Docker using a shared volume.**

When you run the following command, the `-v` argument and path values of `/tmp/models/:/models` will map the `models` folder you created (assuming it was in ) with a folder inside the Docker container. MMS will then be able to use the local model file.

```bash
nvidia-docker run -itd --name mms -p 80:8080 -v /tmp/models/:/models awsdeeplearningteam/mms_gpu "mxnet-model-server start --mms-config /models/mms_app_gpu.conf"
```

**Step 5: Test inference.**

This configuration file is using the default Squeezenet model, so you will request the `squeezenet/predict` API endpoint.

```bash
curl -X POST http://127.0.0.1/squeezenet/predict -F "data=@kitten.jpg"
```

Given that this is a different model, the same image yields a different inference result which will be something similar to the following:

```
{
  "prediction": [
    [
      {
        "class": "n02123159 tiger cat",
        "probability": 0.3630334138870239
      },
...
```

## Reference Commands

Manually pull the MMS Docker CPU image:
```bash
docker pull awsdeeplearningteam/mms_cpu
```

Manually pull the MMS Docker GPU image:
```bash
docker pull awsdeeplearningteam/mms_gpu
```

List your Docker images:
```bash
docker images
```

Verify the Docker container is running:
```bash
docker ps -a
```

Stop the Docker container from running:
```bash
docker rm -f mms
```

Delete the MMS Docker GPU image:
```bash
docker rmi awsdeeplearningteam/mms_gpu
```

Delete the MMS Docker GPU image:
```bash
docker rmi awsdeeplearningteam/mms_cpu
```

Output the recent logs to console.
```bash
docker logs mms
```

Interact with the container. This will open a shell prompt inside the container. Use `$ Ctrl-p-Ctrl-q` to detach again.
```bash
docker attach mms
```

Run the MMS Docker image:
```bash
docker run -itd --name mms -p 80:8080 awsdeeplearningteam/mms_cpu
```

Start MMS in the Docker container (CPU config):
```bash
docker exec mms bash -c "mxnet-model-server start --mms-config /models/mms_app_cpu.conf"
```

Start MMS in the Docker container (GPU config):
```bash
nvidia-docker exec mms bash -c "mxnet-model-server start --mms-config /models/mms_app_cpu.conf"
```

Stop MMS.
```bash
docker exec mms bash -c "mxnet-model-server stop"
```

Get MMS help.
```bash
docker exec mms bash -c "mxnet-model-server help"
```

Refer [Docker CLI](https://docs.docker.com/engine/reference/commandline/run/) to understand each parameter.


## Docker Details

### Docker Hub

Docker images are available on [Docker Hub](https://hub.docker.com/r/awsdeeplearningteam):
* [CPU](https://hub.docker.com/r/awsdeeplearningteam/mms_cpu/)
* [GPU](https://hub.docker.com/r/awsdeeplearningteam/mms_gpu/)
### Building a MMS Docker Image from Scratch
The following are the steps to build a container image from scratch.

#### Prerequisites
In order to build the Docker image yourself you need the following:

* Install Docker
* Clone the MMS repo

#### Docker Installation

For macOS, you have the option of [Docker's Mac installer](https://docs.docker.com/docker-for-mac/install/) or you can simply use `brew`:

```bash
brew install docker
```

For Windows, you should use [their Windows installer](https://docs.docker.com/docker-for-windows/install/).

For Linux, check your favorite package manager if brew is available, otherwise use their installation instructions for [Ubuntu](https://docs.docker.com/engine/installation/linux/ubuntu/) or [CentOS](https://docs.docker.com/engine/installation/linux/centos/).

#### Verify Docker

When you've competed the installation, verify that Docker is running by running `docker images` in your terminal. If this works, you are ready to continue.

#### Clone the MMS Repo

If you haven't already, clone the MMS repo and go into the `docker` folder.

```bash
git clone https://github.com/awslabs/mxnet-model-server.git && cd mxnet-model-server/docker
```

### Building the Container Image

#### Configuration Setup

We can optionally update the **nginx** section of `mms_app_cpu.conf` or `mms_app_gpu.conf` files for your target environment.

* For CPU builds, use [mms_app_cpu.conf](mms_app_cpu.conf) and [Dockerfile.cpu](Dockerfile.cpu).
* For GPU builds, use [mms_app_gpu.conf](mms_app_gpu.conf) and [Dockerfile.gpu](Dockerfile.gpu).

The **nginx** section will look like this:

```text
# Nginx configurations
server {
    listen       8080;

    location / {
        proxy_pass http://unix:/tmp/mms_app.sock;
    }
}
```

The `location` section defines the proxy server to which all the requests are passed. Since **gunicorn** binds to a UNIX socket, a proxy server is the corresponding URL.

#### Configuring the Docker Build for Use on EC2

Now you can examine how to build a Docker image with MMS and establish a public accessible endpoint on EC2 instance. You should be able to adapt this information for any cloud provider. This Docker image can be used in other production environments as well. Skip this section if you're building for local use.

The first step is to create an [EC2 instance](https://aws.amazon.com/ec2/).

### Build Step for CPU container image

There are separate `Dockerfile` configuration files for CPU and GPU. They are named `Dockerfile.cpu` and `Dockerfile.gpu` respectively.

The images are layered in two parts.

1. Base Image - Consists of ubuntu dependenices, gunicorn gevent.
2. MMS Image - Consists of MXNet, MMS and all related python libraries, built on top of the base image

We can build both images, or use prebuilt base image (Hosted on Docker Hub as `awsdeeplearningteam/mms_cpu_base)` and build MMS on top of it.

By default, Docker expects a Dockerfile, so you'll make a copy leaving the original .cpu file as a backup. If you would like to use a GPU instead, follow the separate GPU Build Step further below.
The next command will build the Docker image. The `-t` flag and following value will give the image the tag `mms_image`, however you can specify `mms_image:v0.11` or whatever you want for your tag. If you use just `mms_image`, it will be assigned the default `latest` tag, and be runnable with `mms_image:latest`.

```bash
# Building base image and derived MMS image
docker build -f Dockerfile.cpu.base  awsdeeplearningteam/mms_cpu_base
docker build -f Dockerfile.cpu -t mms_image .
```

```bash
# Building derived MMS image with pre-built base image
docker build -f Dockerfile.cpu -t mms_image .
```

Once this completes, run `docker images` from your terminal. You should see the Docker image listed with the tag, `mms_image:latest`.

### Build Step for GPU

If your host machine has at least one GPU installed, you can use a GPU Docker image to benefit from improved inference performance.

You need to install [nvidia-docker plugin](https://github.com/NVIDIA/nvidia-docker) before you can use a NVIDIA GPU with Docker.

Once you install `nvidia-docker`, run following commands (for info modifying the tag, see the CPU section above):

Similar to CPU base image the prebuilt GPU 'base' image is hosted at `awsdeeplearningteam/mms_gpu_base` (under Docker hub)

```bash
# Building base image and derived MMS image
docker build -f Dockerfile.gpu.base  awsdeeplearningteam/mms_gpu_base
docker build -f Dockerfile.gpu -t mms_image_gpu .
```

```bash
# Building derived  MMS image with pre-built base image
docker build -f Dockerfile.gpu -t mms_image_gpu .
```

#### Running the MMS GPU Docker

```bash
nvidia-docker run -itd -p 80:8080 --name mms -v /home/user/models/:/models mms_image_gpu:latest
```

This command starts the Docker instance in a detached mode and mounts `/home/user/models` of the host system into `/models` directory inside the Docker instance.
Considering that you modified and copied `mms_app_gpu.conf` file into the models directory, before you ran the above `nvidia-docker` command, you would have this configuration file ready to use in the Docker instance.

```bash
nvidia-docker exec mms bash -c "mxnet-model-server start --mms-config /models/mms_app_gpu.conf"
```
You can change the gunicorn argument `--workers` to change utilization of GPU resources. Each worker will utilize one GPU device. Currently up to 4 workers are recommended to get optimal performance for CPU and this should be set to the `number of GPUs` in case of running MXNet Model Server on GPU instances.

### Testing the MMS Docker

Now you can send a request to your server's [api-description endpoint](http://localhost/api-description) to see the list of MMS endpoints or [ping endpoint](http://localhost/ping) to check the health status of the MMS API. Remember to add the port if you used a custom one or the IP or DNS of your server if you configured it for that instead of localhost. Here are some handy test links for common configurations:

* [http://localhost/api-description](http://localhost/api-description)
* [http://localhost:8080/api-description](http://localhost/api-description)
* [http://localhost/ping](http://localhost/ping)

If `mms_app_gpu.conf` or `mms_app_cpu.conf` files are used as is, the following commands can be run to verify that the MXNet Model Server is running.

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1/squeezenet/predict -F "data=@kitten.jpg"
```

The predict endpoint will return a prediction response in JSON. It will look something like the following result:

```json
{
  "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.9408261179924011
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.055966004729270935
      },
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.0025502564385533333
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.00034320182749070227
      },
      {
        "class": "n02123394 Persian cat",
        "probability": 0.00026897044153884053
      }
    ]
  ]
}
```


## Description of Config File Settings

**For mms_app_cpu.conf:**

The system settings are stored in [mms_app_cpu.conf](mms_app_cpu.conf). You can modify these settings to use different models, or to apply other customized settings. The default settings were optimized for a c5.2xlarge instance.

Notes on a couple of the parameters:

* **models** - the model used when setting up service. By default it uses Squeezenet V1.1, change this argument to use customized model.
* **worker-class** - the type of Gunicorn worker processes. We configure by default to `gevent` which is a type of async worker process. Options are described in the [Gunicorn docs](http://docs.gunicorn.org/en/stable/settings.html#worker-class).
* **workers** - the number of Gunicorn workers which gets started. We recommend setting number of workers equal to number of vCPUs in the instance you are using. A detailed discussion of experiments and results can be found [here](../docs/optimised_config.md), if it is left 'optional' MMS will automatically detect vCPUs and assign number of workers to the number of vCPUs. 
* **limit-request-line** - this is a security-related configuration that limits the [length of the request URI](http://docs.gunicorn.org/en/stable/settings.html#limit-request-line). It is useful preventing DDoS attacks.
* **num-gpu** - optional parameter for number of available GPUs user wants to use. MMS currently assigns each guinicorn worker a gpu-id
in the range of 0 .. (num-gpu-1) in a round-robin fashion. **By default MMS uses all the available GPUs but this parameter can be configured if user want to use only few of them**. A discussion on how to set this parameter can be found [here](../docs/optimised_config.md)

```text
    [MMS arguments]
    --models
    squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model

    --service
    optional

    --gen-api
    optional

    --log-file
    optional

    --log-rotation-time
    optional

    --log-level
    optional

    [Gunicorn arguments]
    --bind
    unix:/tmp/mms_app.sock

    --workers
    optional

    ##Following option is used only for GPU and is present in mms_app_gpu.conf
    --num-gpu	     
     optional

    --worker-class
    gevent

    --limit-request-line
    0

    [Nginx configurations]
    server {
        listen       8080;

        location / {
            proxy_pass http://unix:/tmp/mms_app.sock;
        }
    }

    [MXNet environment variables]
    OMP_NUM_THREADS=4
```

## Configuring SSL
`THIS SECTION IS EXPERIMENTAL`

To safely send traffic between the server and the client, you need to setup a secure sockets layer for nginx server.

First of all, you need to get a SSL certificate. It includes a server certificate and a private key. Suppose you have generated the cert (or received one from a CA) and resulting two files are located at /etc/nginx/ssl/nginx.crt and /etc/nginx/ssl/nginx.key.

Second step is to create a Docker container which exposes TCP port 443 for SSL:

```bash
docker run -itd --name mms -v /home/user/models/:/models -p 8080:443 -p 8081:80 mms_image:latest
```

Note that you expose both https and normal http ports.

Third step is to modify nginx section of `mms_app.conf` file to add ssl settings:

```text
server {
    listen       80;
    listen       443 ssl;

    ssl_certificate /etc/nginx/ssl/nginx.crt;
    ssl_certificate_key /etc/nginx/ssl/nginx.key;

    location / {
        proxy_pass http://unix:/tmp/mms_app.sock;
    }
}
```

Now you can try both https://your_public_host_name:8080/ping or http://your_public_host_name:8081/ping to test the service.

```bash
curl -X GET https://your_public_host_name/ping
```

or

```bash
curl -X GET http://your_public_host_name/ping
```
