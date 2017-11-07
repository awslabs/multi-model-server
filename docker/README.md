# DMS with Docker

You will need to build the image yourself. The Docker image is not available on Docker Hub yet.

## Building the Docker Image

The image will be built for CPU, so the underlying deep learning framework will also only use CPU. GPU is not yet supported.

If you haven't already, you will need to [install Docker](https://docs.docker.com/engine/installation). When you've competed the installation, verify that Docker is running by running `docker images` in your terminal. If this works, you are ready to continue.

### Configuration Setup

By default Docker expects a Dockerfile, so we'll make a copy leaving the original .cpu file as a backup.

```bash
    cp Dockerfile.cpu Dockerfile
```

Now let's update the [virtual.conf](virtual.conf) file for your environment. Note the `server_name` entry. You can update `localhost` to be your public hostname, IP address, or just use the default `localhost`. This depends on where you expect to utilize the Docker container.

The `virtual.conf` file will look like this:

```
server {
    listen       80;
    server_name  your_public_host_name;

    location / {
        proxy_pass http://unix:/tmp/dms_app.sock;
    }
}
```

**Note:** You can also modify container's NGINX settings after building and running the Docker container. The config file is located inside the container at `/etc/nginx/conf.d/virtual.conf`, however this will only last while you're running the Docker container and will revert once you shut it down.

### Build Step

```bash
    docker build -t dms_image .
```
You can specify `dms_image:v0.11` or whatever you want for your tag. If you use just `dms_image` it will be assigned the default `latest` tag and be runnable with `dms_image:latest`.

Once this completes, run `docker images` from your terminal. You should see the Docker image listed with the tag, `dms_image:latest`. Skip down to the section **Running the DMS Docker** to continue, or peruse the EC2 instructions if you're curious how to configure it for the cloud.

### Building the Docker Image on EC2

Now we'll go through how to build a Docker image with DMS and establish a public accessible endpoint on EC2 instance. You should be able to adapt this information for any cloud provider. This Docker image can be used in other production environments as well.

The first step is to create an (EC2 instance)[https://aws.amazon.com/ec2/].
Before we start to build the Docker image, change the `server_name` entry in the [virtual.conf](virtual.conf) file to be your public hostname: `localhost` should be replaced with the public DNS of the EC2 instance.

TODO: this seems to be missing the security settings steps needed to poke a hole in the instance's firewall

## Running the DMS Docker

Since we used a general tag, `dms_image` we're go to run it using the default tag assignment of `dms_image:latest`. Or, if you used a specific tag, make sure you swap that out for the Docker ID, or how your image appears when your run `docker images`.

You may also want to modify the `-p 80:80` to utilize other ports instead. Refer to [Docker documentation on managing ports on hosts](https://docs.docker.com/engine/userguide/networking/default_network/dockerlinks/#connect-using-network-port-mapping) for more info.

```bash
    docker run -it -p 80:80 dms_image:latest
```
Now that you have a prompt inside the container, the final step is to setup DMS endpoint, gunicorn wsgi entry point, and nginx proxy_pass. Using the Docker container's bash prompt, run the following command:

```bash
    cd dms_docker_cpu && ./launch.sh
```

At this point you should see the typical DMS CLI output indicating that the server is running.

## System Settings  

The system settings are stored in `dms_docker_cpu/dms_app.config`. You can modify these settings to use different models, or to apply other customized settings. The default settings were optimized for a C4.8xlarge instance.

    # deep-model-server arguments
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

    # Gunicorn arguments
    --bind
    unix:/tmp/dms_app.sock

    --workers
    4

    --worker-class
    gevent

    --limit-request-line
    0

    # MXNet environment variables
    OMP_NUM_THREADS=4

## Testing the DMS Docker

Now you can send a request to http://your_public_host_name/api-description to see the list of DMS endpoints or http://your_public_host_name/ping to check the health status of the DMS API.

## Use docker image for gpu

If your host machine has at least one GPU installed, you can use GPU docker image to benefit from improved inference performance.

You need to install [nvidia-docker plugin](https://github.com/NVIDIA/nvidia-docker) before you can use nvidia gpu inside docker.

Once you install nvidia-docker, run following commands:

```bash
cp Dockerfile.gpu Dockerfile
docker build -t dms_image_gpu .
nvidia-docker run -it -p 80:80 dms_image_gpu:latest
```

Now you are inside docker container and dms config file is localted in dms_docker_gpu folder. Run following command to launch service:

```bash
cd dms_docker_gpu && ./launch.sh
```
You can change gunicorn argument `--workers` to change utilization of gpu resources. Each worker would utilize one gpu device. Currently up to 4 workers are recommended to get optimal performance.
