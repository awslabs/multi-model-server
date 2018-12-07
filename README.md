Model Server for Apache MXNet
=======

| ubuntu/python-2.7 | ubuntu/python-3.6 |
|---------|---------|
| ![Python3 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoicGZ6dXFmMU54UGxDaGsxUDhXclJLcFpHTnFMNld6cW5POVpNclc4Vm9BUWJNamZKMGdzbk1lOU92Z0VWQVZJTThsRUttOW8rUzgxZ2F0Ull1U1VkSHo0PSIsIml2UGFyYW1ldGVyU3BlYyI6IkJJaFc1QTEwRGhwUXY1dDgiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) | ![Python2 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiYVdIajEwVW9uZ3cvWkZqaHlaRGNUU2M0clE2aUVjelJranJoYTI3S1lHT3R5THJXdklzejU2UVM5NWlUTWdwaVVJalRwYi9GTnJ1aUxiRXIvTGhuQ2g0PSIsIml2UGFyYW1ldGVyU3BlYyI6IjArcHVCaFgvR1pTN1JoSG4iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) |

Model Server for Apache MXNet (MMS) is a flexible and easy to use tool for serving deep learning models exported from [MXNet](http://mxnet.io/) or the Open Neural Network Exchange ([ONNX](http://onnx.ai/)).


Use the MMS Server CLI, or the pre-configured Docker images, to start a service that sets up HTTP endpoints to handle model inference requests.

A quick overview and examples for both serving and packaging are provided below. Detailed documentation and examples are provided in the [docs folder](docs/README.md).

Join our [<img src='docs/images/slack.png' width='20px' /> slack channel](https://join.slack.com/t/mms-awslabs/shared_invite/enQtNDk4MTgzNDc5NzE4LTBkYTAwMjBjMTVmZTdkODRmYTZkNjdjZGYxZDI0ODhiZDdlM2Y0ZGJiZTczMGY3Njc4MmM3OTQ0OWI2ZDMyNGQ) to get in touch with development team, ask questions, find out what's cooking and more!

## Contents of this Document
* [Quick Start](#quick-start)
* [Serve a Model](#serve-a-model)
* [Other Features](#other-features)
* [External demos powered by MMS](#external-demos-powered-by-mms)
* [Contributing](#contributing)


## Other Relevant Documents
* [Latest Version Docs](docs/README.md)
* [v0.4.0 Docs](https://github.com/awslabs/mxnet-model-server/blob/v0.4.0/docs/README.md)
* [Migrating from v0.4.0 to v1.0.0](docs/migration.md)

## Quick Start
### Prerequisites
Before proceeding further with this document, make sure you have the following prerequisites.
1. Ubuntu, CentOS, or macOS. Windows support is experimental. The following instructions will focus on Linux and macOS only.
1. Python     - MXNet model server requires python to run the workers.
1. pip        - Pip is a python package management system.
1. Java 8     - MXNet Model Server requires Java 8 to start. You have the following options for installing Java 8:

    For Ubuntu:
    ```bash
    sudo apt-get install openjdk-8-jre-headless
    ```

    For CentOS:
    ```bash
    sudo yum install java-1.8.0-openjdk
    ```

    For macOS:
    ```bash
    brew tap caskroom/versions
    brew update
    brew cask install java8
    ```

### Installing MXNet Model Server with pip

#### Setup

**Step 1:** Setup a Virtual Environment

We recommend installing and running MXNet Model Server in a virtual environment. It's a good practice to run and install all of the Python dependencies in virtual environments. This will provide isolation of the dependencies and ease dependency management.

One option is to use Virtualenv. This is used to create virtual Python environments. You may install and activate a virtualenv for Python 2.7 as follows:

```bash
pip install virtualenv
```

Then create a virtual environment:
```bash
# Assuming we want to run python2.7 in /usr/local/bin/python2.7
virtualenv -p /usr/local/bin/python2.7 /tmp/pyenv2
# Enter this virtual environment as follows
source /tmp/pyenv2/bin/activate
```

Refer to the [Virtualenv documentation](https://virtualenv.pypa.io/en/stable/) for further information.

**Step 2:** Install MXNet
MMS won't install the MXNet engine by default. If it isn't already installed in your virtual environment, you must install one of the MXNet pip packages.

For CPU inference, `mxnet-mkl` is recommended. Install it as follows:

```bash
# Recommended for running MXNet Model Server on CPU hosts
pip install mxnet-mkl
```

For GPU inference, `mxnet-cu92mkl` is recommended. Install it as follows:

```bash
# Recommended for running MXNet Model Server on GPU hosts
pip install mxnet-cu92mkl
```

**Step 3:** Install or Upgrade MMS as follows:

```bash
# Install latest released version of mxnet-model-server 
pip install mxnet-model-server
```

To upgrade from a previous version of `mxnet-model-server`, please refer [migration reference](docs/migration.md) document.

**Notes:**
* A minimal version of `model-archiver` will be installed with MMS as dependency. See [model-archiver](model-archiver/README.md) for more options and details.
* See the [advanced installation](docs/install.md) page for more options and troubleshooting.

### Serve a Model

Once installed, you can get MMS model server up and running very quickly. Try out `--help` to see all the CLI options available.

```bash
mxnet-model-server --help
```

For this quick start, we'll skip over most of the features, but be sure to take a look at the [full server docs](docs/server.md) when you're ready.

Here is an easy example for serving an object classification model:
```bash
mxnet-model-server --start --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

With the command above executed, you have MMS running on your host, listening for inference requests. **Please note, that if you specify model(s) during MMS start - it will automatically scale backend workers to the number equal to available vCPUs (if you run on CPU instance) or to the number of available GPUs (if you run on GPU instance). In case of powerful hosts with a lot of compute resoures (vCPUs or GPUs) this start up and autoscaling process might take considerable time. If you would like to minimize MMS start up time you can try to avoid registering and scaling up model during start up time and move that to a later point by using corresponding [Management API](docs/management_api.md#register-a-model) calls (this allows finer grain control to how much resources are allocated for any particular model).**

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
To stop the current running model-server instance, run the following command:
```bash
$ mxnet-model-server --stop
```
You would see output specifying that mxnet-model-server has stopped.

### Create a Model Archive

MMS enables you to package up all of your model artifacts into a single model archive. This makes it easy to share and deploy your models.
To package a model, check out [model archiver documentation](model-archiver/README.md)

## Recommended production deployments

* MMS doesn't provide authentication. You have to have your own authentication proxy in front of MMS.
* MMS doesn't provide throttling, it's vulnerable to DDoS attack. It's recommended to running MMS behind a firewall.
* MMS only allows localhost access by default, see [Network configuration](docs/configuration.md#configure-mms-listening-port) for detail.
* SSL is not enabled by default, see [Enable SSL](docs/configuration.md#enable-ssl) for detail.
* MMS use a config.properties file to configure MMS's behavior, see [Manage MMS](docs/configuration.md) page for detail of how to configure MMS.
* For better security, we recommend running MMS inside docker container. This project includes Dockerfiles to build containers recommended for production deployments. These containers demonstrate how to customize your own production MMS deployment. The basic usage can be found on the [Docker readme](docker/README.md).

## Other Features

Browse over to the [Docs readme](docs/README.md) for the full index of documentation. This includes more examples, how to customize the API service, API endpoint details, and more.

## External demos powered by MMS

Here are some example demos of deep learning applications, powered by MMS:

 |  |   |
|:------:|:-----------:|
| [Product Review Classification](https://thomasdelteil.github.io/TextClassificationCNNs_MXNet/) <img width="325" alt="demo4" src="https://user-images.githubusercontent.com/3716307/48382335-6099ae00-e695-11e8-8110-f692b9ecb831.png"> |[Visual Search](https://thomasdelteil.github.io/VisualSearch_MXNet/) <img width="325" alt="demo1" src="https://user-images.githubusercontent.com/3716307/48382332-6099ae00-e695-11e8-9fdd-17b5e7d6d0ec.png">|
| [Facial Emotion Recognition](https://thomasdelteil.github.io/FacialEmotionRecognition_MXNet/) <img width="325" alt="demo2" src="https://user-images.githubusercontent.com/3716307/48382333-6099ae00-e695-11e8-8bc6-e2c7dce3527c.png"> |[Neural Style Transfer](https://thomasdelteil.github.io/NeuralStyleTransfer_MXNet/) <img width="325" alt="demo3" src="https://user-images.githubusercontent.com/3716307/48382334-6099ae00-e695-11e8-904a-0906cc0797bc.png"> |

## Contributing

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.
