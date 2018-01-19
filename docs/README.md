# MXNet Model Server Documentation

## Basic Features
* [Serving Quick Start](../README.md#serve-a-model) - Basic server usage tutorial
* [Exporting Quick Start](../README.md#export-a-model) - Basic export usage tutorial
* [Installation](install.md) - Installation and troubleshooting
* [Serving Models](server.md) - How to use `mxnet-model-server`
  * [REST API](rest_api.md) - Specification on the API endpoint for MMS
  * [Model Zoo](model_zoo.md) - A collection of MMS .model files that you can use with MMS
* [Exporting Models](export.md) - How to use `mxnet-model-export`
    * [Export Examples](export_examples.md)
    * [Export a Model Checkpoint from MXNet](export_from_mxnet.md)
    * [Export an ONNX Model](export_from_onnx.md)
    * [Exported MMS Model File Tour](export_model_file_tour.md)
* [Docker](../docker/README.md) - How to use MMS with Docker and cloud services
* [Metrics](metrics.md) - How to configure logging of metrics

## Advanced Features
* [Custom Model Service](custom_service.md) - Custom inference services and dev guide for implementation
* [Client Code Generation](code_gen.md) - Use Swagger to create a client API for over 40 languages; a JavaScript example is provided
* [Unit Tests](../mms/tests/README.md) - Housekeeping unit tests for MMS
* [Load Test](../load-test/README.md) - Use JMeter to run MMS through the paces and collect load data

## Example Projects
* [LSTM](../examples/lstm_ptb/README.md) - An example MMS project for an RNN / LSTM that will take json inputs for inference against a model trained with a specific vocabulary
* [Object Detection](../examples/ssd/README.md) - An example MMS project for using a pre-trained Single Shot Multi Object Detection (SSD) model taking image inputs and inferring the types and locations of several classes of objects
