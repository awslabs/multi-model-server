# Migration from MMS 0.4

MMS 1.0 is a major release that contains significant architecture improvement based on MMS 0.4.

MMS 1.0 adopted micro-services based architecture, the frontend request handler is separated from backend inference worker. The frontend is a java based web service which provide REST API, and backend is python based worker which execute custom service code (Other language worker support is also planed).

## Table of Content

* [Installation](#installation)
* [Command line interface](#command-line-interface)
* [API](#api)
* [Model archive](#model-archive)
* [Docker container](#docker-container)
* [Logging](#logging)
* [Metrics](#metrics)
* [Configuration](#configuration)
    * [SSL](#ssl)

## Installation

MMS 1.0 made following changes for pip installation package:

* **java 8**: java is required for MMS 1.0. You must install java 8 (or later) and make sure java is on available in $PATH environment variable *before* installing MMS. If you have multiple java installed, you can use $JAVA_HOME environment vairable to control which java to use.
* **mxnet**: `mxnet` will not be installed by default with MMS 1.0 any more. You have to install it manually.

See more detail: [Install MMS](install.md)

## Command line interface
MMS 1.0 made some parameter changes in `mxnet-model-server` command line tool. The old command line parameters won't work any more.

Please see [Command Line Interface](server.md#command-line-interface)

## API
You can continue to use MMS 0.4 inference API in MMS 1.0. However they are deprecated. Please migrate to new [inference API](inference_api.md)

## Model archive
You can continue to use your existing MMS 0.4 model archive (`.model` file). We stronger recommend you to migrate to new Model archive (`.mar`) format.

Please refer to following documents:
* [Custom service code](custom_service.md)
* [model-archiver tool](../model-archiver/README.md)
* [Create model archive example](../examples/mxnet_vision/README.md)

### model-archiver
`mxnet-model-export` is no longer supported. Instead we release a `model-archiver` CLI tool. `model-archiver` now can be installed standalone:

```bash
pip install model-archiver
```
See [model-archiver](../model-archiver/README.md) for more detail.

## Docker container

MMS docker image makes it easier for you to serve a model. In 0.4 release, MMS require a configuration file (mms_app_cpu.conf or mms_app_gpu.conf) to start MMS in docker container. The old conf file format is no longer supported. To make it simple, MMS no longer requires the --mms-config parameter, the default configuration should work for most of use cases. MMS will start automatically while docker container starts:

```bash
docker run -itd --name mms -p 80:8080 -p 8081:8081 awsdeeplearningteam/mms_cpu
```

After docker container started, you can use [Management API](management_api.md) to load models for inference.

See [Docker Image](../docker/README.md) for detail.

## Logging

MMS 1.0 provides highly customizable logging feature. MMS 0.4 logging parameter (--log-file, , --log-rotation-time and --log-level) in command line is not supported.

For more detail see [Logging configuration](logging.md)

## Metrics

MMS 1.0 redesigned metrics feature:
* The old --metrics-write-to parameter is not supported, instead a rich configuration is provided.
* The built-in ClouldWatch integration is removed, instead MMS 1.0 provide a template allows user to integrated with any metrics server.

See [Metrics](metrics.md) for more detail.

## Configuration

MMS 1.0 provide a rich set of configuration parameters allow advanced user to customize/tune MMS. A completely new set of parameters are introduced in new config.properties file. The MMS 0.4 format of configuration file is not supported any more.

See [Advanced configuration](configuration.md) for more detail.

### SSL

MMS 0.4 support SSL via nginx, now MMS 1.0 provide native SSL support. See [Enable SSL](configuration.md#enable-ssl) for detail.
