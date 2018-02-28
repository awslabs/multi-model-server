
# Install MMS

## Prerequisites

* **Python**: Required. Model Server for Apache MXNet (MMS) works with Python 2 or 3.  When installing MMS, we recommend that you use a Python and Conda environment to avoid conflicts with your other Apache MXNet or Open Neural Network Exchange (ONNX) installations.
* **protoc**: Optional. If you plan to use ONNX features, you need to install the [protobuf compiler](https://github.com/onnx/onnx#installation). Install it *before* installing MMS.

* **Curl**: Optional. Curl is used in all of the examples. Install it with your preferred package manager.

* **Unzip**: Optional. Unzip allows you to easily extract model files and inspect their content. If you choose to use it, associate it with `.model` extensions.

## Install MMS with pip

To install MMS for the first time, install Python, then run the following command:

```bashpip install mxnet-model-server```

To upgrade from a previous version of MMS, run:

```bashpip install -U mxnet-model-server```

## Install MMS from Source Code

If you prefer, you can clone MMS from source code. First, run the following command:

```bashgit clone https://github.com/awslabs/mxnet-model-server.git && cd mxnet-model-server```

To install MMS, run:

```bashpip install .```

To upgrade MMS, run:

```bashpip install -U .```

## Install MMS for Development

If you plan to develop with MMS and change some of the source code, install it from source code and make your changes executable with this command:

```bashpip install -e .```

To upgrade MMS from source code and make changes executable, run:

```bashpip install -U -e .```

## Troubleshooting Installation

| Issue | Platform | Solution |
|---|---|---|
|Could not find "protoc" executable! | Ubuntu |Run `sudo apt-get install protobuf-compiler libprotoc-dev`. |
|| macOS | Run `conda install -c conda-forge protobuf`.
|Missing [LibGFortran](https://gcc.gnu.org/onlinedocs/gfc-internals/LibGFortran.html) library| Ubuntu | Run `apt-get install libgfortran3`. |
|| Amazon Linux |Run `yum install gcc-gfortran`. |
