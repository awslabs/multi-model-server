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

Now let's update the nginx section of [dms_app.config](dms_docker_cpu/dms_app.conf) file for your environment. Note the `server_name` entry. You can update `localhost` to be your public hostname, IP address, or just use the default `localhost`. This depends on where you expect to utilize the Docker container.

The nginx section will look like this:

```
server {
    listen       80;
    server_name  your_public_host_name;

    location / {
        proxy_pass http://unix:/tmp/dms_app.sock;
    }
}
```

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

Notes on a couple of the parameters:

* **worker-class** - the type of Gunicorn worker processes. We configure by default to gevent which is a type of async worker process. Options are [described in the Gunicorn docs](http://docs.gunicorn.org/en/stable/settings.html#worker-class).
* **limit-request-line** - this is a security-related configuration that limits the [length of the request URI](http://docs.gunicorn.org/en/stable/settings.html#limit-request-line). It is useful preventing DDoS attacks.

```
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

    # Nginx configurations
    server {
        listen       80;
        server_name  localhost;

        location / {
            proxy_pass http://unix:/tmp/dms_app.sock;
        }
    }

    # MXNet environment variables
    OMP_NUM_THREADS=4
```

## Testing the DMS Docker

Now you can send a request to http://your_public_host_name/api-description to see the list of DMS endpoints or http://your_public_host_name/ping to check the health status of the DMS API.

## Use Docker Image for GPU

If your host machine has at least one GPU installed, you can use a GPU docker image to benefit from improved inference performance.

You need to install [nvidia-docker plugin](https://github.com/NVIDIA/nvidia-docker) before you can use a NVIDIA GPU with Docker.

Once you install `nvidia-docker`, run following commands:

```bash
cp Dockerfile.gpu Dockerfile
docker build -t dms_image_gpu .
nvidia-docker run -it -p 80:80 dms_image_gpu:latest
```

Now you should be inside the docker container at a bash prompt. The config file, `dms_app.config` is located in the `dms_docker_gpu` folder. Run following command to launch the DMS service:

```bash
cd dms_docker_gpu && ./launch.sh
```
You can change the gunicorn argument `--workers` to change utilization of GPU resources. Each worker will utilize one GPU device. Currently up to 4 workers are recommended to get optimal performance.

## Configuring HTTPS servers

To safely send traffic between the server and the client, we need to setup a secure sockets layer for nginx server.

First of all, we need to get a SSL certificate. It includes a server certificate and a private key. Suppose you have generated the cert (or received one from a CA) and resulting two files are located at /etc/nginx/ssl/nginx.crt and /etc/nginx/ssl/nginx.key.

Second step is to create a docker container which exposes TCP port 443 for SSL:

```bash
docker run -it -p 8080:443 -p 8081:80 dms_image:latest
```

Note that we expose both https and normal http ports.

Third step is to modify nginx section of `dms_app.config` file to add ssl settings:

```
server {
    listen       80;
    listen       443 ssl;

    server_name  your_public_host_name;
    ssl_certificate /etc/nginx/ssl/nginx.crt;
    ssl_certificate_key /etc/nginx/ssl/nginx.key;

    location / {
        proxy_pass http://unix:/tmp/dms_app.sock;
    }
}
```

Launch service by typing:
```bash
./launch.sh
```

Now we can try both https://your_public_host_name:8080/ping or http://your_public_host_name:8081/ping to test the service.
