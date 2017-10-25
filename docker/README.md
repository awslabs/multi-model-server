## Docker Usage
We'll go through how to build a DMS cpu docker image and establish a public accessible endpoint on EC2 instance. This docker image can be used in other production environments as well.

First step is to create an (EC2 instance)[https://aws.amazon.com/ec2/].
Before we start to build docker image, change the server_name entry in virtual_config to be your public hostname:

    server {
        listen       80;
        server_name  your_public_host_name;

        location / {
            proxy_pass http://unix:/tmp/dms_app.sock;
        }
    }

For EC2 instance, this should be public DNS of instance.
You can also modify this config file after building and running a docker container. It is located at /etc/nginx/conf.d/virtual.conf

Second step is to build docker image.:
```bash
    cp Dockerfile.cpu Dockerfile
    docker build -t mms_image .
```

Third step is to run a docker container:
```bash
    docker run -it -p 80:80 mms_image:latest
```
Final step is to setup DMS endpoint, gunicorn wsgi entry point and nginx proxy_pass:
Inside docker container, run following command:
```bash
    cd dms_docker && ./launch.sh
```
System settings are stored in dms_app.config. You can modify these settings to use different models or apply other customized settings.

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
    1

    --worker-class
    gevent

    --limit-request-line
    0

Now you can send request to http://your_public_host_name.
In browser, type in `http://your_public_host_name/ping` to check health status.