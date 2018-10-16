# Model archiver for MMS

## Contents of this Document
* [Overview](#overview)
* [Model Archiver CLI](#model-archiver-command-line-interface)
* [Artifact Details](#artifact-details)
    * [MAR-INFO](#mar-inf)
    * [Model name](#model-name)
    * [Runtime](#runtime)
    * [Handler](#handler)

## Other Relevant Documents
* [Model Archive Example](../examples/mxnet_vision/README.md)
* [Packaging an ONNX Model](docs/convert_from_onnx.md)

## Overview

A key feature of MMS is the ability to package all model artifacts into a single model archive file. It is a separate command line interface (CLI), `model-archiver`, that can take model checkpoints and package them into a `.mar` file that can then be redistributed and served by anyone using MMS. It takes in the following model artifacts: a model composed of one or more files, the description of the model's inputs in the form of a signature file, a service file describing how to handle inputs and outputs, and other optional assets that may be required to serve the model. The CLI creates a `.mar` file that MMS's server CLI uses to serve the models.

**Important**: Make sure you try the [Quick Start: Creating a Model Archive](../README.md#create-a-model-archive) tutorial for a short example of using `model-archiver`.

MMS support arbitrary models file. It's custom service code's responsibility to locate and load models files. Following information are required to create a standalone model archive:
1. [Model name](#model-name)
2. [Model path](#model-path)
3. [Handler](#handler)

## Model Archiver Command Line Interface

Now let's cover the details on using the CLI tool: `model-archiver`.

Example usage with the squeezenet_v1.1 model archive you may have downloaded or created in the [main README's](../README.md) examples:

```bash

model-archiver --model-name squeezenet_v1.1 --model-path squeezenet --handler mxnet_vision_service:handle

```

### Arguments

```
$ model-archiver -h
usage: model-archiver [-h] --model-name MODEL_NAME --model-path MODEL_PATH
                      --handler HANDLER [--runtime {python,python2,python3}]
                      [--export-path EXPORT_PATH] [-f]

Model Archiver Tool

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Exported model name. Exported file will be named as
                        model-name.mar and saved in current working directory
                        if no --export-path is specified, else it will be
                        saved under the export path
  --model-path MODEL_PATH
                        Path to the folder containing model related files.
  --handler HANDLER     Handler path to handle custom MMS inference logic.
  --runtime {python,python2,python3}
                        The runtime specifies which language to run your
                        inference code on. The default runtime is
                        RuntimeType.PYTHON. At the present moment we support
                        the following runtimes python, python2, python3
  --export-path EXPORT_PATH
                        Path where the exported .mar file will be saved. This
                        is an optional parameter. If --export-path is not
                        specified, the file will be saved in the current
                        working directory.
  -f, --force           When the -f or --force flag is specified, an existing
                        .mar file with same name as that provided in --model-
                        name in the path specified by --export-path will
                        overwritten
```

## Artifact Details

### MAR-INF
**MAR-INF** is reserved folder name that will be used inside `.mar` file. This folder contains model archive metadata files. User should avoid using **MAR-INF** in the model path.

### Runtime

### Model name

A valid model name. The model name must begin with a letter of the alphabet, and can only contains letters, digits, underscore (_), dash (-) and dot (.)

**Note**: The model name can be override when register model with [Register Model API](../ docs/management_api.md#register-a-model).

### Model path

A folder that contains all necessary files that need to run inference code for the model. All the files and sub-folders (except [excluded files](#excluded-files)) will be packaged into `.mar` file.

#### excluded files
Follow type of file will be excluded during model archive packaging:
1. hidden files
2. Mac system files: __MACOSX and .DS_Store
3. MANIFEST.json file
4. python compiled byte code (.pyc) files and cache folder __pycache__

### handler

A handler is an python entry point that MMS can invoke to executes inference code. The format of a Python handler is:
* python_module_name[:function_name] (for example: lstm-service:handle).

The function name is optional if provided python module follows one of predefined convention:
1. There is a `handle()` function available in the module
2. The module contains only one Class and the class that contains a `handle()` function.

Further details and specifications are found on the [custom service](../docs/custom_service.md) page.

