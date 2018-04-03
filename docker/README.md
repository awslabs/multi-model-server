# Using Containers with MXNet Model Server

MXNet Model Server (MMS) can be used with any container service. In this guide, you will learn how to run MMS with Docker.

## Contents of this Document
* [Quickstart](#quickstart)
* [Configuring MMS with Docker](#configuring-mms-with-docker)


## Other Relevant Documents
* [Advanced Settings](advanced_settings.md)
    * [GPU Inference](advanced_settings.md#gpu-inference)
    * [Reference Commands](advanced_settings.md#reference-commands)
    * [Docker Details](advanced_settings.md#docker-details)
    * [Description of Config File Settings](advanced_settings.md#description-of-config-file-settings)
    * [Configuring SSL](advanced_settings.md#configuring-ssl)
* [Launch MMS as a managed inference service on AWS Fargate](../docs/mms_on_fargate.md)
    * [Introduction to published containers](../docs/mms_on_fargate.md#familiarize-yourself-with-our-containers)
    * [Creating a AWS Fargate task to server SqueezeNet V1.1](../docs/mms_on_fargate.md#create-a-aws-faragte-task-to-serve-squeezenet-model)
    * [Creating an Load Balancer](../docs/mms_on_fargate.md#create-a-load-balancer)
    * [Creating an AWS ECS Service](../docs/mms_on_fargate.md#creating-an-ecs-service-to-launch-our-aws-fargate-task)
    * [Testing your service](../docs/mms_on_fargate.md#test-your-service)
    * [Build custom MMS containers images to serve your Deep learning models](../docs/mms_on_fargate.md#customize-the-containers-to-server-your-custom-deep-learning-models)

## Quickstart
Running MXNet Model Server with Docker in two steps:

**Step 1: Run the Docker image.**

This will download the MMS Docker image and run its default configuration, serving a Squeezenet model.

```bash
docker run -itd --name mms -p 80:8080 awsdeeplearningteam/mms_cpu mxnet-model-server start --mms-config /mxnet_model_server/mms_app_cpu.conf
```

With the `-p` flag, we're setting it up so you can run inference on your host computer's port: `80`.

**Step 2: Test inference.**

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1/squeezenet/predict -F "data=@kitten.jpg"
```

After fetching this image of a kitten and posting it to the `predict` endpoint, you should see a response similar to the following:

```
{
  "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.9408261179924011
...
```

### Cleaning Up

Now that you have tested it out, you can stop the Docker container. The following command with stop the server and delete the container, but retain the Docker image for trying out other models and configurations next.

```bash
docker rm -f mms
```

## Configuring MMS with Docker

In the Quickstart section, you launched a Docker image with MMS serving the Squeezenet model. Now you will learn how to configure MMS with Docker to run other models, as well as how to collect MMS logs, and optimize your MMS with Docker images.

### Using MMS and Docker with a Shared Volume

For the purpose of loading different models and retrieving logs from MMS you will setup a shared volume with the Docker image.


**Step 1: Create a folder to share with the Docker container.**

Create a directory for `models`. This will also provide a place for log files to be written.

```bash
mkdir /tmp/models
```

**Step 2: Download the configuration template.**

To run a different model in the Docker image, you need to provide it a new configuration file.

Download the template for a CPU or a GPU config and place it in the `models` folder you just created:
* [mms_app_cpu.conf](mms_app_cpu.conf)
* [mms_app_gpu.conf](mms_app_gpu.conf)

**Step 3: Modify the configuration template.**

Edit the file you downloaded, `mms_app_*pu.conf`. It will have the following section for `MMS Arguments`:

```
[MMS Arguments]
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
```

To change the model, you will update the `--models` argument. This uses MMS's flexible handling of the model file locations, so that the model can reside at a URL, or on your local file system.

Change `--models` to use the following `resnet-18` model:

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
docker run -itd --name mms -p 80:8080 -v /tmp/models/:/models awsdeeplearningteam/mms_cpu
```

**Step 5: Start MMS.**
You also need to start MMS in the container. In the Quickstart, we did this as all one command, but here you are doing it as a second step.

```bash
docker exec mms bash -c "mxnet-model-server start --mms-config /models/mms_app_cpu.conf"
```

**Step 6: Test inference.**

You will upload the same kitten image as before, but this time you will request the `resnet-18/predict` API endpoint.

```bash
curl -X POST http://127.0.0.1/resnet-18/predict -F "data=@kitten.jpg"
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

Now that you have tried the default inference using Squeezenet and configuring inference to the resnet-18 model you are ready to try some other more advanced settings.

* GPU inference
* MMS settings
* Optimizing inference with gunicorn configurations

Next Step: [Advanced Settings](advanced_settings.md)