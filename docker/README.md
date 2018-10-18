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

This will download the MMS Docker image and run its default configuration, serving a SqueezeNet model.

```bash
docker run -itd --name mms -p 80:8080 -p 81:8081 awsdeeplearningteam/mms_cpu mxnet-model-server --start --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

With the `-p` flag, we're setting it up so you can run the Predict API on your host computer's port `80`. This maps to the Docker image's port `8080`.
It will run the Management API on your host computer's port `81`. This maps to the Docker image's port `8081`.

**Step 2: Test inference.**

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1/predictions/squeezenet -T @kitten.jpg
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

Now that you have tested it out, you may stop the Docker container. The following command will stop the server and delete the container. It will retain the Docker image for trying out other models and configurations later.

```bash
docker rm -f mms
```

## Configuring MMS with Docker

In the Quickstart section, you launched a Docker image with MMS serving the SqueezeNet model.
Now you will learn how to configure MMS with Docker to run other models.
You will also learn how to collect MMS logs, and optimize MMS with Docker images.

### Using MMS and Docker with a Shared Volume

You may sometimes want to load different models with a different configuration.
Setting up a shared volume with the Docker image is the recommended way to handle this.

**Step 1: Create a folder to share with the Docker container.**

Create a directory for `models`. This will also provide a place for log files to be written.

```bash
mkdir /tmp/models
```

**Step 2: Download the configuration template.**

Download the template `config.properties` and place it in the `models` folder you just created:
* [config.properties](config.properties)

**Step 3: Modify the configuration template.**

Edit the file you downloaded, `config.properties`.

```properties
# vmargs=-Xmx1g -XX:MaxDirectMemorySize=512m -Dlog4j.configuration=file:///opt/ml/conf/log4j.properties
model_store=/models
# load_models=ALL
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
# number_of_netty_threads=0
# max_workers=0
# job_queue_size=1000
# number_of_gpu=1
# keystore=src/test/resources/keystore.p12
# keystore_pass=changeit
# keystore_type=PKCS12
# private_key_file=src/test/resources/key.pem
# certificate_file=src/test/resources/certs.pem
```

Modify the configuration file to suite your configuration needs before running the model server.

Save the file.

**Step 4: Run MMS with Docker using a shared volume.**

When you run the following command, the `-v` argument and path values of `/tmp/models/:/models` will map the Docker image's `models` folder to your local `/tmp/models` folder.
MMS will then be able to use the local model file.

```bash
docker run -itd --name mms -p 80:8080 -p 81:8081 -v /tmp/models/:/models awsdeeplearningteam/mms_cpu mxnet-model-server --start --mms-config /models/config.properties --models resnet=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model
```

**NOTE**: If you modify the inference_address or the management_address in the configuration file,
you must modify the ports exposed by Docker as well.

**Step 5: Test inference.**

You will upload the same kitten image as before, but this time you will request the `predictions/resnet` API endpoint.

```bash
curl -X POST http://127.0.0.1/predictions/resnet -T @kitten.jpg
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

## Conclusion

You have tried the default Predictions API settings using a SqueezeNet model. 
You then configured your Predictions API endpoints to also serve a ResNet-18 model.
Now you are ready to try some other more **advanced settings** such as:

* GPU inference
* MMS settings
* Optimizing inference with gunicorn configurations

Next Step: [Advanced Settings](advanced_settings.md)
