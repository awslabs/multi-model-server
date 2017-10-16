## Docker Usage
We'll go through how to build a DMS cpu docker image and establish an endpoint proxy-passed by nginx.

First step is to build docker image:
```bash
    cp dockerfile.cpu Dockerfile
    docker build -t mms_image .
```

Second step is to run a docker container:
```bash
    docker run -it -p 80:80 mms_image:latest
```

Third step is to setup DMS endpoint and nginx proxy_pass:
Inside docker container, run following command:
```bash
    service nginx start
    deep-model-server --models resnet-18=https://s3.amazonaws.com/mms-models/resnet-18.model
```
Now you can send request to http://localhost from outside docker container.
In browser, type in `http://localhost/ping` to check health status.