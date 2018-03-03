# Model Server or Apache MXNet Documentation

## Basic Features
* [Serving Quick Start](../README.md#serve-a-model) - Basic server usage tutorial
* [Exporting Quick Start](../README.md#export-a-model) - Tutorial that shows you how to export.
* [Installation](install.md) - Installation procedures and troubleshooting
* [Serving Models](server.md) - Explains how to use `mxnet-model-server`.
  * [REST API](rest_api.md) - Specification on the API endpoint for MMS
  * [Model Zoo](model_zoo.md) - A collection of MMS .model files that you can use with MMS.
* [Exporting Models](export.md) - Explains how to use `mxnet-model-export`.
    * [Export Examples](export_examples.md)
    * [Export an ONNX Model](export_from_onnx.md)
    * [Exported MMS Model File Tour](export_model_file_tour.md)
* [Docker](../docker/README.md) - How to use MMS with Docker and cloud services
* [Metrics](metrics.md) - How to configure logging of metrics

## Advanced Features
* [Custom Model Service](custom_service.md) - Describes how to develop custom inference services.
* [Client Code Generation](code_gen.md) - Shows how to use Swagger to create a client API for more than 40 languages. Includes a JavaScript example.
* [Unit Tests](../mms/tests/README.md) - Housekeeping unit tests for MMS
* [Load Test](../load-test/README.md) - Use JMeter to run MMS through the paces and collect load data

## Example Projects
* [LSTM](../examples/lstm_ptb/README.md) - An example MMS project for a recurrent neural network (RNN) using long short-term memory (LSTM). The project takes JSON inputs for inference against a model trained with a specific vocabulary.
* [Object Detection](../examples/ssd/README.md) - An example MMS project that uses a pretrained Single Shot Multi Object Detection (SSD) model that takes image inputs and infers the types and locations of several classes of objects.
