# Deep Model Server

The primary feature of DMS is the model server. It can be used for many types of inference in production settings. It provides an easy-to-use command line interface and utilizes an industry standard OpenAPI interface. It has features for pre-processing and post-processing inputs and outputs for customized applications.

While current models and support is limited to MXNet as the underlying framework, other framework support and serving of models originating from other frameworks will be supported in the future.

## Command Line Interface

```bash
$ deep-model-server
usage: deep-model-server [-h] --models KEY1=VAL1,KEY2=VAL2...
                         [KEY1=VAL1,KEY2=VAL2... ...] [--service SERVICE]
                         [--gen-api GEN_API] [--port PORT] [--host HOST]
                         [--gpu]
                         [--log-file LOG_FILE]
                         [--log-rotation-time LOG_ROTATION_TIME]
                         [--log-level LOG_LEVEL]
```

### Required Arguments & Defaults

Example single model usage:

```bash
deep-model-server --models name=model_location
```

Example multiple model usage:

```bash
deep-model-server --models name=model_location, name2=model_location2
```

`--models` is the only required argument. You can pass one or more models in a key value pair format: `name` you want to call the model and `model_location` for the local file path or URI to the model. The name is what appears in your REST API's endpoints. In the first example we used `squeezenet_v1.1` for the name, e.g. `deep-model-server --models squeezenet_v1.1=...`, and accordingly the predict endpoint was called by `http://127.0.0.1:8080/squeezenet_v1.1/predict`. In the first example this was `squeezenet=https://s3.amazonaws.com/mms-models/squeezenet_v1.1.model`. Alternatively, we could have downloaded the file and used a local file path like `squeezenet=dms_models/squeezenet_v1.1.model`.

The rest of these arguments are optional and will have the following defaults:
* [--service mxnet_vision_service]
* [--port 8080]
* [--host 127.0.0.1]

gpu argument is to specifiy whether to use gpu for inference.

Logging and exporting an SDK can also be triggered with additional arguments. Details are in the following Arguments section.

#### Arguments:
1. **models**: required, <model_name>=<model_path> pairs.

    (a) Model path can be a local file path or URI (s3 link, or http link).
        local file path: path/to/local/model/file or file://root/path/to/model/file
        s3 link: s3://S3_endpoint[:port]/...
        http link: http://hostname/path/to/resource

    (b) Currently, the model file has .model extension, it is actually a zip file with a .model extension packing trained MXNet models and model signature files. The details will be explained in **Export existing model** section

    (c) Multiple models loading are also supported by specifying multiple name path pairs
2. **service**: optional, the system will load input service module and will initialize MXNet models with the service defined in the module. The module should contain a valid class which extends the base model service with customized `_preprocess` and `_postprocess` functions.
3. **port**: optional, default is 8080
4. **host**: optional, default is 127.0.0.1
5. **gpu**: optional, gpu device id, such as 0 or 1. cpu will be used if this argument is not set.
5. **gen-api**: optional, this will generate an open-api formated client sdk in build folder.
6. **log-file**: optional, log file name. By default it is "dms_app.log".
7. **log-rotation-time**: optional, log rotation time. By default it is "1 H", which means one hour. Valid format is "interval when". For weekday and midnight, only "when" is required. Check https://docs.python.org/2/library/logging.handlers.html#logging.handlers.TimedRotatingFileHandler for detail values.
8. **log-level**: optional, log level. By default it is INFO. Possible values are NOTEST, DEBUG, INFO, ERROR AND CRITICAL. Check https://docs.python.org/2/library/logging.html#logging-levels


## Serving Multiple Models with DMS

Here's an example for running the resnet-18 and the vgg16 models using local model files.

```bash
deep-model-server --models resnet-18=file://models/resnet-18 vgg16=file://models/vgg16
```

This will setup a local host serving resnet-18 model and vgg16 model on the same port, using the default 8080.

Note that if you supply a custom service for pre or post-processing, both models will use that same pipeline. There is currently no support for using different pipelines per-model.
