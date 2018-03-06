# MMS with Docker

You will need to build the image yourself. The Docker image is not available on Docker Hub yet.

## Prerequisites

In order to build the Docker image yourself you need the following:

* Install Docker
* Clone the MMS repo

### Docker Installation

For Mac you the option of [their Mac installer](https://docs.docker.com/docker-for-mac/install/) or you can simply use brew:

```bash
brew install docker
```

For Windows, you should use [their Windows installer](https://docs.docker.com/docker-for-windows/install/).

For Linux, check your favorite package manager if brew is available, otherwise use their installation instructions for [Ubuntu](https://docs.docker.com/engine/installation/linux/ubuntu/) or [CentOS](https://docs.docker.com/engine/installation/linux/centos/).

#### Verify Docker

When you've competed the installation, verify that Docker is running by running `docker images` in your terminal. If this works, you are ready to continue.

### Clone the MMS Repo

If you haven't already, clone the MMS repo and go into the `docker` folder.

```bash
git clone https://github.com/awslabs/mxnet-model-server.git && cd mxnet-model-server/docker
```


## Building the Docker Image

### Configuration Setup

We can optionally update the **nginx** section of the `mms_app_*.conf` file for your target environment. If you're going to run the Docker image locally you can leave this alone and skip to the **Build Step**. If you want to run it on a publicly accessible IP or DNS name then continue with this setup step.

* For CPU builds, use [mms_app_cpu.conf](mms_app_cpu.conf) and [Dockerfile.cpu](Dockerfile.cpu).
* For GPU builds, use [mms_app_gpu.conf](mms_app_gpu.conf) and [Dockerfile.gpu](Dockerfile.gpu).

Note the `server_name` entry. You can update `localhost` to be your public hostname, IP address, or just use the default `localhost`. This depends on where you expect to utilize the Docker image. (Server Name can be updated at run-time. 
This option is mentioned in steps to run.)

The **nginx** section will look like this:

```
# Nginx configurations
server {
    listen       80;
    server_name  localhost;

    location / {
        proxy_pass http://unix:/tmp/mms_app.sock;
    }
}
```
The `location` section defines the proxy server to which all the requests are passed. Since **gunicorn** binds to a UNIX socket, proxy server is the corresponding URL.

### Configuring the Docker Build for Use on EC2

Now we'll go through how to build a Docker image with MMS and establish a public accessible endpoint on EC2 instance. You should be able to adapt this information for any cloud provider. This Docker image can be used in other production environments as well. Skip this section if you're building for local use.

The first step is to create an (EC2 instance)[https://aws.amazon.com/ec2/].
Before we start to build the Docker image, change the `server_name` entry of `nginx` configuration section in the [mms_app.conf](mms_docker_cpu/mms_app.conf) file to be your public hostname: `localhost` should be replaced with the public DNS of the EC2 instance.

## Build Step for CPU

There are separate `Dockerfile` configuration files for CPU and GPU. They are named `Dockerfile.cpu` and `Dockerfile.gpu` respectively.

By default Docker expects a Dockerfile, so we'll make a copy leaving the original .cpu file as a backup. If you would like to use a GPU instead, follow the separate GPU Build Step further below.
The next command will build the Docker image. The `-t` flag and following value will give the image the tag `mms_image`, however you can specify `mms_image:v0.11` or whatever you want for your tag. If you use just `mms_image` it will be assigned the default `latest` tag and be runnable with `mms_image:latest`.

```bash
docker build -f Dockerfile.cpu -t mms_image .
```

Once this completes, run `docker images` from your terminal. You should see the Docker image listed with the tag, `mms_image:latest`. Skip down to the section **Running the MMS Docker** to continue, or peruse the EC2 instructions if you're curious how to configure it for the cloud.

## Build Step for GPU

If your host machine has at least one GPU installed, you can use a GPU docker image to benefit from improved inference performance.

You need to install [nvidia-docker plugin](https://github.com/NVIDIA/nvidia-docker) before you can use a NVIDIA GPU with Docker.

Once you install `nvidia-docker`, run following commands (for info modifying the tag, see the CPU section above):

```bash
docker build -f Dockerfile.gpu -t mms_image_gpu .
```

## Preparing `models` and `mms_app.conf` files for running the Model Server
Create a `models` directory on the host machine and add the models to be used along with the mms_app_[cpu|gpu].conf file into the directory. Modify the `mms_app[cpu/gpu].conf` file to reflect the model files to be used along with updated options for other `gunicorn`, `nginx` and `MMS` configurations.
 ```bash
 # Modify the mms_app_cpu.conf or mms_app_gpu.conf and add it to this folder
 $ mkdir models
 $ cp ~/mxnet-model-server/docker/mms_app_cpu.conf models/
 ```
## Running the MMS Docker

Since we used a general tag, `mms_image` we're go to run it using the default tag assignment of `mms_image:latest`. Or, if you used a specific tag, make sure you swap that out for the Docker ID, or how your image appears when your run `docker images`.

You may also want to modify the `-p 80:80` to utilize other ports instead. Refer to [Docker documentation on managing ports on hosts](https://docs.docker.com/engine/userguide/networking/default_network/dockerlinks/#connect-using-network-port-mapping) for more info.

**Note**: if you're using the GPU Docker, skip ahead to the next section.

```bash
# Run the docker image in a detached mode
$ docker run -itd -p 80:80 --name mms -v /home/user/models:/models mms_image:latest
```

To run the MMS with $HOSTNAME as its endpoint, or for `nginx's server_name` to be configured to `$HOSTNAME` run the following command.
```bash
# Start docker with nginx's server_name configured to $HOSTNAME
$ docker run -itd -p 80:80 --name mms -v /home/user/models:/models -e MXNET_MODEL_SERVER_HOST=$HOSTNAME mms_image:latest
```
The above command lets you run inference with `$HOSTNAME` as 'server_name'.  
   
Verify that this image is running by running 
```bash
$ docker ps -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                NAMES
a6141053ef51        mms_cpu             "/bin/bash"         15 minutes ago      Up 15 minutes       0.0.0.0:80->80/tcp   mms
$
```
You should see that the image is running.

## Starting MMS in the docker instance
To interact with the docker image named `mms` we run the following command
```bash
$ docker exec mms bash -c "mxnet-model-server.sh help"
Usage:

/mxnet_model_server/mxnet-model-server.sh [start | stop | restart | help] [--mms-config <MMS config file>]

start        : Start a new instance of MxNet model server.
stop         : Stop the current running instance of MxNet model server
restart      : Restarts all the MMS worker instances.
help         : Usage help for /mxnet_model_server/mxnet-model-server.sh
--mms-config : Location pointing to the MxNet model server configuration file.
To start the MxNet model server, run
/mxnet_model_server/mxnet-model-server.sh start --mms-config <path-to-config-file\>

To stop the running instance of MxNet model server, run
/mxnet_model_server/mxnet-model-server.sh stop

To restart the running instance of MxNet model server, run
/mxnet_model_server/mxnet-model-server.sh restart --mms-config <path-to-config-file\>
```

Verify that the docker image is still running by running 
```bash
$ docker ps -a
```

```bash
# To start the MMS run the following
$ docker exec mms bash -c "mxnet-model-server.sh start --mms-config /models/mms_app_cpu.conf"
```

This will setup the MMS endpoint, gunicorn wsgi entry point, and nginx proxy_pass. 
At this point you should be able to run inference on `localhost` port `80`

### Running the MMS GPU Docker

```bash
$ nvidia-docker run -itd -p 80:80 --name mms -v /home/user/models/:/models mms_image_gpu:latest
```

To configure the nginx hostname to the $HOSTNAME, run the following command
```bash
# To run inference, use the $HOSTNAME instead of 'localhost' or '127.0.0.1' to ping. This Hostname can be the public DNS/IP. 
$ nvidia-docker run -itd -p 80:80 --name mms -v /home/user/models:/models -e MXNET_MODEL_SERVER_HOST=$HOSTNAME mms_image_gpu:latest
```

This command starts the docker instance in a detached mode and mounts `/home/user/models` of the host system into `/models` directory inside the Docker instance. 
Considering that you modified and copied `mms_app_gpu.conf` file into the models directory, before you ran the above `nvidia-docker` command, you would have this configuration file ready to use in the docker instance.

```bash
$ nvidia-docker exec mms bash -c "mxnet-model-server.sh start --mms-config /models/mms_app_gpu.conf"
```
You can change the gunicorn argument `--workers` to change utilization of GPU resources. Each worker will utilize one GPU device. Currently up to 4 workers are recommended to get optimal performance.

## Testing the MMS Docker

Now you can send a request to your server's [api-description endpoint](http://localhost/api-description) to see the list of MMS endpoints or [ping endpoint](http://localhost/ping) to check the health status of the MMS API. Remember to add the port if you used a custom one or the IP or DNS of your server if you configured it for that instead of localhost. Here are some handy test links for common configurations:

* [http://localhost/api-description](http://localhost/api-description)
* [http://localhost:8080/api-description](http://localhost:8080/api-description)
* [http://localhost/ping](http://localhost/ping)

## Advanced Settings

### Description of Settings in mms_app.conf

The system settings are stored in `mms_docker_cpu/mms_app.conf`. You can modify these settings to use different models, or to apply other customized settings. The default settings were optimized for a C4.8xlarge instance.

Notes on a couple of the parameters:

* **models** - the model used when setting up service. By default it uses resnet-18, chnage this argument to use customized model.
* **worker-class** - the type of Gunicorn worker processes. We configure by default to gevent which is a type of async worker process. Options are [described in the Gunicorn docs](http://docs.gunicorn.org/en/stable/settings.html#worker-class).
* **limit-request-line** - this is a security-related configuration that limits the [length of the request URI](http://docs.gunicorn.org/en/stable/settings.html#limit-request-line). It is useful preventing DDoS attacks.

```
    [MMS arguments]
    --models
    resnet-18=https://s3.amazonaws.com/mms-models/resnet-18.model

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
    4

    --worker-class
    gevent

    --limit-request-line
    0

    [Nginx configurations]
    server {
        listen       80;
        server_name  localhost;

        location / {
            proxy_pass http://unix:/tmp/mms_app.sock;
        }
    }

    [MXNet environment variables]
    OMP_NUM_THREADS=4
```

### Configuring SSL for HTTPS

To safely send traffic between the server and the client, we need to setup a secure sockets layer for nginx server.

First of all, we need to get a SSL certificate. It includes a server certificate and a private key. Suppose you have generated the cert (or received one from a CA) and resulting two files are located at /etc/nginx/ssl/nginx.crt and /etc/nginx/ssl/nginx.key.

Second step is to create a docker container which exposes TCP port 443 for SSL:

```bash
docker run -itd --name mms -v /home/user/models/:/models -p 8080:443 -p 8081:80 mms_image:latest
```

Note that we expose both https and normal http ports.

Third step is to modify nginx section of `mms_app.conf` file to add ssl settings:

```
server {
    listen       80;
    listen       443 ssl;

    server_name  your_public_host_name;
    ssl_certificate /etc/nginx/ssl/nginx.crt;
    ssl_certificate_key /etc/nginx/ssl/nginx.key;

    location / {
        proxy_pass http://unix:/tmp/mms_app.sock;
    }
}
```

Now we can try both https://your_public_host_name:8080/ping or http://your_public_host_name:8081/ping to test the service.

```bash
curl -X GET https://your_public_host_name:80/ping
```

or

```bash
curl -X GET http://your_public_host_name:80/ping
```

## Stopping the current MMS instance
To stop the MMS running inside the docker instance, run the following
```bash
$ docker exec mms bash -c "mxnet-model-server.sh stop"
```
## Debugging or logging into the running docker instance
To debug the docker instance further you could run the following commands

#### docker logs mms
This provides the console logs from the latest run of the MMS command.

#### docker attach mms
This attaches the `detached` docker running instance and since we run the docker instance in interactive mode, we will land up in console. If a version of MMS is running, you would have to kill it with `Ctrl-c` or put it in the background with `Ctrl-z` to get to the console.
To exit the docker instance without quitting, type `Ctrl-p-Ctrl-q`. **Do not quit by typing `exit` as this will exit the docker instance**.
```bash
# To come out of an attached docker instance
$ Ctrl-p-Ctrl-q
```

#### docker rm -f mms
If you find any issue with the current running instance (named mms) of docker, you could kill it by running the above command.

For other useful CLI commands, please visit [Docker docs](https://docs.docker.com/edge/engine/reference/commandline/docker/)