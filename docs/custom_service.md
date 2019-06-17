# Custom Service

## Contents of this Document
* [Introduction](#introduction)
* [Requirements for custom service file](#requirements-for-custom-service-file)
* [Example Custom Service file](#example-custom-service-file)
* [Creating model archive with entry point](#creating-model-archive-with-entry-point)

## Introduction

A custom service , is the code that is packaged into model archive, that is executed by Model Server for Apache MXNet (MMS). 
The custom service is responsible for handling incoming data and passing on to engine for inference. The output of the custom service is returned back as response by MMS.

## Requirements for custom service file

The custom service file should define a method that acts as an entry point for execution, this function will be invoked by MMS on a inference request. 
The function can have any name, not necessarily handle, however this function should accept, the following parameters
    
* **data** - The input data from the incoming request
* **context** - Is the MMS [context](https://github.com/vrakesh/mxnet-model-server/blob/master/mms/context.py) information passed for use with the custom service if required. 


The signature of a entry point function is:

```python
def function_name(data,context):
    """
    Works on data and context passed
    """
    # Use parameters passed
```
The next section, showcases an example custom service.

## Example Custom Service file

```python
# custom service file

# model_handler.py

"""
ModelHandler defines a base model handler.
"""
import logging


class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        return None

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        return None

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return ["OK"] * self._batch_size
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

```
Here the ``` handle()``` method is our entry point that will be invoked by MMS, with the parameters data and context, it in turn can pass this information to an actual inference class object or handle all the processing in the 
```handle()``` method itself. The ```initialize()``` method is used to initialize the model at load time, so after first time, the service need not be re-initialized in the the life cycle of the relevant worker.
 We recommend using a ```initialize()``` method, avoid initialization at prediction time.
 
 This entry point is engaged in two cases: (1) when MMS is asked to scale a model up, to increase the number of backend workers (it is done either via a ```PUT /models/{model_name}``` request or a ```POST /models``` request with `initial-workers` option or during MMS startup when you use `--models` option (```mxnet-model-server --start --models {model_name=model.mar}```), ie., you provide model(s) to load) or (2) when MMS gets a ```POST /predictions/{model_name}``` request. (1) is used to scale-up or scale-down workers for a model. (2) is used as a standard way to run inference against a model. (1) is also known as model load time, and that is where you would normally want to put code for model initialization. You can find out more about these and other MMS APIs in [MMS Management API](./management_api.md) and [MMS Inference API](./inference_api.md)

## Creating model archive with entry point 

MMS, identifies the entry point to the custom service, from the manifest file. Thus file creating the model archive, one needs to mention the entry point using the ```--handler``` option. 

The [model-archiver](https://github.com/awslabs/mxnet-model-server/blob/master/model-archiver/README.md) tool enables the create to an archive understood by MMS.

```python
model-archiver --model-name <model-name> --handler model_handler:handle --export-path <output-dir> --model-path <model_dir> --runtime python3
```

This will create file ```<model-name>.mar``` in the directory ```<output-dir>```

This will create a model archive with the custom handler, for python3 runtime. The ```--runtime``` parameter enables usage of specific python version at runtime, by default it uses the default python distribution of the system.
