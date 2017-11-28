# Running the Model Server

The primary feature of MMS is the model server. It can be used for many types of inference in production settings. It provides an easy-to-use command line interface and utilizes an industry standard [OpenAPI interface](rest_api.md). It has features for pre-processing and post-processing inputs and outputs for customized applications.

For example, you want to make an app that lets your users snap a picture, and it'll tell them what what objects were detected in the scene and predictions on what the objects might be. You can use MMS to serve a prediction endpoint for a object detection and identification model that intakes images, then returns predictions. You can also modify MMS behavior with custom services and run multiple models. There are examples of custom services, pre-processing, post-processing in the [examples](../examples) folder. The object detection example is in [examples/ssd](../examples/ssd/README.md).

## Technical Details

Now that you have a high level view of MMS, let's get a little into the weeds. MMS takes a deep learning model and it wraps it in a REST API. Currently it is bundled with the MXNet framework, and it comes with a built-in web server that you run from command line. This command line call takes in the single or multiple models you want to serve, along with additional optional parameters controlling the port, host, and logging. Additionally, you can point it to service extensions which define pre-processing and post-processing steps. MMS also comes with a default vision service that makes it easy to serve an image classification model. If you're looking to build chat bots or video understanding then you'll have some additional leg work to do with the pre-processing and post-processing steps. These are covered in more detail in the [custom service](custom_service.md) documentation.

To try out MMS serving now, you can load the SqueezeNet model, which is under 5 MB, with this example:

```bash
mxnet-model-server --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model --service mms/model_service/mxnet_vision_service.py
```

With the command above executed, you have MMS running on your host, listening for inference requests.

To test it out, download a [cute picture of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and name it `kitten.jpg`. Then run the following `curl` command to post an inference request with the image. In the example below both of these steps are provided.

```bash
wget -O kitten.jpg \
  https://upload.wikimedia.org/wikipedia/commons/8/8f/Cute-kittens-12929201-1600-1200.jpg
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "data=@kitten.jpg"
```

Then check the [API description](http://127.0.0.1:8080/api-description). For info on other endpoints check out the [REST API documentation](rest_api.md).

For more models, check out the [model zoo](model_zoo.md).

To learn about serving different kinds of model and inference types, take a look at [custom services](custom_service.md).

## Model Files

The rest of this topic focus on serving of model files without much discussion on the model files themselves, where they come from, and how they're made. Long story short: it's a zip archive with the parameters, weights, and metadata that define a model that has been trained already. If you want to know more about the model files, take a look at the [export tool documentation](export.md).

## Command Line Interface

```bash
$ mxnet-model-server --help
usage: mxnet-model-server [-h] --models KEY1=VAL1 KEY2=VAL2...
                          [KEY1=VAL1 KEY2=VAL2... ...] [--service SERVICE]
                          [--gen-api GEN_API] [--port PORT] [--host HOST]
                          [--gpu GPU] [--log-file LOG_FILE]
                          [--log-rotation-time LOG_ROTATION_TIME]
                          [--log-level LOG_LEVEL]
                          [--metrics-write-to {log,csv}]

MXNet Model Server

optional arguments:
  -h, --help            show this help message and exit
  --models KEY1=VAL1 KEY2=VAL2... [KEY1=VAL1 KEY2=VAL2... ...]
                        Models to be deployed using name=model_location
                        format. Location can be a URL or a local path to a
                        .model file. Name is arbitrary and used as the API
                        endpoints base name.
  --service SERVICE     Path to a user defined model service.
  --gen-api GEN_API     Generates API client for the supplied language.
                        Options include Java, C#, JavaScript and Go. For
                        complete list check out https://github.com/swagger-api
                        /swagger-codegen.
  --port PORT           Port number. By default it is 8080.
  --host HOST           Host. By default it is localhost.
  --gpu GPU             ID of GPU device to use for inference. If your machine
                        has N gpus, this number can be 0 to N - 1. If it is
                        not set, cpu will be used.
  --log-file LOG_FILE   Log file name. By default it is "mms_app.log".
  --log-rotation-time LOG_ROTATION_TIME
                        Log rotation time. By default it is "1 H", which means
                        one hour. Valid format is "interval when", where
                        _when_ can be "S", "M", "H", or "D". For a particular
                        weekday use only "W0" - "W6". For midnight use only
                        "midnight". Check https://docs.python.org/2/library/lo
                        gging.handlers.html#logging.handlers.TimedRotatingFile
                        Handler for detailed information on values.
  --log-level LOG_LEVEL
                        Log level. By default it is INFO. Possible values are
                        NOTEST, DEBUG, INFO, ERROR AND CRITICAL. Check
                        https://docs.python.org/2/library/logging.html
                        #logging-levelsfor detailed information on values.
  --metrics-write-to {log,csv}
                        Target location to write MMS metrics. Log file
                        specified in --log-file or to local CSV files per
                        metric type.
```

### Required Arguments & Defaults

Example single model usage:

```bash
mxnet-model-server --models name=model_location
```

`--models` is the only required argument. You can pass one or more models in a key value pair format: `name` you want to call the model and `model_location` for the local file path or URI to the model. The name is what appears in your REST API's endpoints. In the first example we used `squeezenet_v1.1` for the name, e.g. `mxnet-model-server --models squeezenet_v1.1=...`, and accordingly the predict endpoint was called by `http://127.0.0.1:8080/squeezenet_v1.1/predict`. In the first example this was `squeezenet=https://s3.amazonaws.com/mms-models/squeezenet_v1.1.model`. Alternatively, we could have downloaded the file and used a local file path like `squeezenet=mms_models/squeezenet_v1.1.model`.

The rest of these arguments are optional and will have the following defaults:
* [--port 8080]
* [--host 127.0.0.1]
* [--log-file mms_app.log]
* [--log-rotation-time "1 H"] - one hour
* [--log-level INFO]
* [--metrics-write-to log] - will write to `mms_app.log`

Advanced logging, GPU-based inference, and exporting an SDK can also be triggered with additional arguments. Details are in the following Arguments section.

#### Arguments:
1. **models**: required, <model_name>=<model_path> pairs.

    a) Model path can be a local file path or URI (s3 link, or http link).
        local file path: path/to/local/model/file or file://root/path/to/model/file
        s3 link: s3://S3_endpoint[:port]/...
        http link: http://hostname/path/to/resource

    b) Currently, the model file has .model extension, it is actually a zip file with a .model extension packing trained MXNet models and model signature files. The details will be explained in **Export existing model** section.

    c) Multiple models loading are also supported by specifying multiple name path pairs.
1. **service**: optional, path to a custom service module. More info in the Custom Services section.
1. **port**: optional, default is 8080
1. **host**: optional, default is 127.0.0.1
1. **gpu**: optional, GPU device id, such as 0 or 1. cpu will be used if this argument is not set.
1. **gen-api**: optional, this will generate an open-api formatted client SDK in build folder. More information is in the generate API section.
1. **log-file**: optional, log file name. By default it is "mms_app.log". More info is in the logging section.
1. **log-rotation-time**: optional, log rotation time. By default it is "1 H", which means one hour. More info is in the logging section.
1. **log-level**: optional, log level. By default it is INFO. More info is in the logging section.
1. **metrics-write-to**: optional, metrics location/style. By default it is log. More info is in the logging section.

## Advanced Features

### Inference with GPU

The `gpu` argument is to specify whether to use a GPU for inference. Currently there is no batching capability in MMS, so multiple GPUs are not supported. This argument is the ID, starting with 0, of the GPU device you want to use.

### Custom Names

You can change the name of the model prediction endpoint to be whatever you want. In our previous examples we used `squeezenet_v1.1=<url>`, but we can shorten that or use whatever name we want. For example, let's use `squeezenet` instead:

```bash
mxnet-model-server --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

### Custom Services

This topic is covered in much more detail on the [custom service documentation page](custom_service.md), but let's talk about how you start up your MMS server using a custom service and why you might want one.
Let's say you have a model named `super-fancy-net.model` that can detect a lot of things, but you want an API endpoint that detects only hotdogs. You would use a name that makes sense for it, such as the "not-hot-dog" API. In this case we might invoke MMS like this:

```bash
mxnet-model-server --models not-hot-dog=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/super-fancy-net.model
```

You would also want to customize and limit MMS inference with a custom service, put that code into a Python file (e.g. nothotdog.py) along with the model file, and call that script with the `--service` argument as in this example:

```bash
mxnet-model-server --models not-hot-dog=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/super-fancy-net.model --service nothotdog.py
```

This would serve a prediction endpoint at `/not-hot-dog/predict` and run your custom service code that is located in `nothotdog.py`. For more info on custom services, check out the [object detection example](../examples/ssd/README.md) and the [custom service documentation](custom_service.md).

### Serving Multiple Models with MMS

Example multiple model usage:

```bash
mxnet-model-server --models name=model_location name2=model_location2
```

Here's an example for running the resnet-18 and the vgg16 models using local model files.

```bash
mxnet-model-server --models resnet-18=file://models/resnet-18 squeezenet=file://models/squeezenet_v1.1
```

If you don't have the model files locally, then you can call MMS using URLs to the model files.

```bash
mxnet-model-server --models resnet=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

This will setup a local host serving resnet-18 model and squeezenet model on the same port, using the default 8080. Check http://127.0.0.1:8080/api-description to see that each model has an endpoint for prediction. In this case you would see `resnet/predict` and `squeezenet/predict`/

Note that if you supply a [custom service](custom_service.md) for pre or post-processing, both models will use that same pipeline. There is currently no support for using different pipelines per-model.

### Logging Features

The are four arguments for MMS that facilitate logging of the model serving and inference activity.

1. **log-file**: optional, log file name. By default it is "mms_app.log". You may also specify a path and a custom file name such as `logs/squeezenet_inference`. This is the root file name that is used in file rotation.

1. **log-rotation-time**: optional, log rotation time. By default it is "1 H", which means one Hour. Valid format is "interval when", where _when_ can be "S", "M", "H", or "D". For a particular weekday use only "W0" - "W6". For midnight use only "midnight". When a file is rotated a timestamp is appended, for example, `squeezenet_inference` would look like `squeezenet_inference.2017-11-27_17-26` after log rotation. Check the [Python docs on logging handlers](https://docs.python.org/2/library/logging.handlers.html#logging.handlers.TimedRotatingFileHandler) for detailed information on values.

1. **log-level**: optional, log level. By default it is INFO. Possible values are NOTEST, DEBUG, INFO, ERROR and CRITICAL. Check the [Python docs for logging levels](https://docs.python.org/2/library/logging.html#logging-levels) for more information.

1. **metrics-write-to**: various server metrics are gathered and are written to the default log file, but if the `csv` value is passed to this argument, the metrics are recorded every 30 seconds in separate CSV files as follows.

      a) **mms_cpu.csv** - CPU load

      b) **mms_errors.csv** - number of errors

      c) **mms_memory.csv**	- memory utilization

      d) **mms_preprocess_latency.csv** - any custom pre-processing latency

      e) **mms_disk.csv** - disk utilization

      f) **mms_inference_latency.csv** - any inference latency

      g) **mms_overall_latency.csv** - collective latency

      h) **mms_requests.csv** - number of inference requests

### Client API Code Generation

Detailed info on using the `gen-api` argument and its outputs is found on the [Code Generation page](code_gen.md). 
