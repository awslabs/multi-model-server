Model Server for Apache MXNet
=======

| ubuntu/python-2.7 | ubuntu/python-3.5 |
|---------|---------|
| ![Python3 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoicGZ6dXFmMU54UGxDaGsxUDhXclJLcFpHTnFMNld6cW5POVpNclc4Vm9BUWJNamZKMGdzbk1lOU92Z0VWQVZJTThsRUttOW8rUzgxZ2F0Ull1U1VkSHo0PSIsIml2UGFyYW1ldGVyU3BlYyI6IkJJaFc1QTEwRGhwUXY1dDgiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) | ![Python2 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiYVdIajEwVW9uZ3cvWkZqaHlaRGNUU2M0clE2aUVjelJranJoYTI3S1lHT3R5THJXdklzejU2UVM5NWlUTWdwaVVJalRwYi9GTnJ1aUxiRXIvTGhuQ2g0PSIsIml2UGFyYW1ldGVyU3BlYyI6IjArcHVCaFgvR1pTN1JoSG4iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) |

Apache MXNet Model Server (MMS) is a flexible and easy to use tool for serving deep learning models exported from [MXNet](http://mxnet.io/) or the Open Neural Network Exchange ([ONNX](http://onnx.ai/)).


Use the MMS Server CLI, or the pre-configured Docker images, to start a service that sets up HTTP endpoints to handle model inference requests.

A quick overview and examples for both serving and packaging are provided below. Detailed documentation and examples are provided in the [docs folder](docs/README.md).

## Contents of this Document
* [Quick Start](#quick-start)
* [Serve a Model](#serve-a-model)
* [Create a Model Archive](#create-a-model-archive)
* [Other Features](#other-features)
* [Contributing](#contributing)

## Other Relevant Documents
* [Latest Version Docs](docs/README.md)
* [v0.4.0 Docs](https://github.com/awslabs/mxnet-model-server/blob/v0.4.0/docs/README.md)
## Quick Start

### Install with pip

A minimal version of `model-archiver` will be installed with MMS as dependency. See [model-archiver](model-archiver/README.md) for more options and detail.

MMS runtime depends on Python and java-8, please make sure install Python and java 8 (or later)  before install MMS.

For ubuntu:
```bash
sudo apt-get install openjdk-8-jre-headless
```

For centos:
```bash
sudo yum install java-1.8.0-openjdk
```

For Mac:
```bash
brew tap caskroom/versions
brew update
brew cask install java8
```

MMS won't install mxnet engine by default, you can install mxnet-mkl or mxnet-cu90mkl based on your need.

```bash
pip install mxnet-mkl

pip install -U mxnet-model-server
```

See the [advanced installation](docs/install.md) page for more options and troubleshooting.

### Serve a Model

Once installed, you can get MMS model serving up and running very quickly. Try out `--help` to see the kind of features that are available.

```bash
mxnet-model-server --help
```

For this quick start, we'll skip over most of the features, but be sure to take a look at the [full server docs](docs/server.md) when you're ready.

Here is an easy example for serving an object classification model:
```bash
mxnet-model-server --start --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

With the command above executed, you have MMS running on your host, listening for inference requests.

To test it out, you can open a new terminal window next to the one running MMS. Then you can use `curl` to download one of these [cute pictures of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and curl's `-o` flag will name it `kitten.jpg` for you. Then you will `curl` a `POST` to the MMS predict endpoint with the kitten's image.

![kitten](docs/images/kitten_small.jpg)

In the example below, we provide a shortcut for these steps.

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/predictions/squeezenet -T kitten.jpg
```

The predict endpoint will return a prediction response in JSON. It will look something like the following result:

```json
[
  {
    "probability": 0.8582232594490051,
    "class": "n02124075 Egyptian cat"
  },
  {
    "probability": 0.09159987419843674,
    "class": "n02123045 tabby, tabby cat"
  },
  {
    "probability": 0.0374876894056797,
    "class": "n02123159 tiger cat"
  },
  {
    "probability": 0.006165083032101393,
    "class": "n02128385 leopard, Panthera pardus"
  },
  {
    "probability": 0.0031716004014015198,
    "class": "n02127052 lynx, catamount"
  }
]
```

You will see this result in the response to your `curl` call to the predict endpoint, and in the server logs in the terminal window running MMS. It's also being [logged locally with metrics](docs/metrics.md).

Other models can be downloaded from the [model zoo](docs/model_zoo.md), so try out some of those as well.

Now you've seen how easy it can be to serve a deep learning model with MMS! [Would you like to know more?](docs/server.md)

### Stopping the running model server
To stop the current running model-server instance, you could run the following command
```bash
$ mxnet-model-server --stop
```
You would see an output specifying that the model-server running instance stopped.

### Create a Model Archive

MMS enables you to package up all of your model artifacts into a single model archive, that you can then easily share or distribute. To package a model, follow these **three** steps:

**1. Download sample squeezenet modle artifacts (if you don't have them handy)**

```bash
mkdir squeezenet
cd squeezenet

curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-symbol.json
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-0000.params
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/signature.json
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt
```

The downloaded model artifact files are:

* **Model Definition** (json file) - contains the layers and overall structure of the neural network
* **Model Params and Weights** (params file) - contains the parameters and the weights
* **Model Signature** (json file) - defines the inputs and outputs that MMS is expecting to hand-off to the API
* **assets** (text files) - auxiliary files that support model inference such as vocabularies, labels, etc. and vary depending on the model

Further details on these files, custom services, and advanced `model-archiver` features can be found on the [Package Models for Use with MMS](model-archiver/README.md) page.

**2. Prepare your model custom service code**

You can implement your own model customer service code as model archive entry point. Here we are going to use provided mxnet vision service template:

```bash
cp -r mxnet-model-server/examples/template/* squeezenet/
``` 

**3. Package Your Model**

With the model artifacts available locally, you can use the `model-archiver` CLI to generate a `.mar` file that can be used to serve an inference API with MMS.

In this next step we'll run `model-archiver` and tell it our model's prefix is `squeezenet_v1.1` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets.

**Note**: For mxnet models, `model-name` must match prefix of the symbol and param file name. 

```bash
model-archiver --model-name squeezenet_v1.1 --model-path squeezenet --handler mxnet_vision_service:handle
```

This will package all the model artifacts files in `squeezenet` directory and output `squeezenet_v1.1.mar` in the current working directory. This `.mar` file is all you need to run MMS, serving inference requests for a simple image recognition API. Go back to the Serve a Model tutorial above and try to run this model archive that you just created!

To learn more about `model-archiver`, check out [Model archiver documentation](model-archiver/README.md)

## Recommended production deployments

* MMS doesn't provide authentication. You have to your own authentication proxy in front of MMS.
* MMS doesn't provide throttling, it's vulnerable to DDoS attack. It's recommended to running MMS behind a firewall.
* MMS only allows localhost access by default, see [Network configuration](docs/configuration.md#configure-mms-listening-port) for detail.
* SSL is not enabled by default, see [Enable SSL](docs/configuration.md#enable-ssl) for detail.
* MMS use a config.properties file to configure MMS's behavior, see [Manage MMS](docs/configuration.md) page for detail of how to configure MMS.
* For better security, we recommend running MMS inside docker container. This project includes Dockerfiles to build containers recommended for production deployments. These containers demonstrate how to customize your own production MMS deployment. The basic usage can be found on the [Docker readme](docker/README.md).

## Other Features

Browse over to the [Docs readme](docs/README.md) for the full index of documentation. This includes more examples, how to customize the API service, API endpoint details, and more.

## Contributing

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.
