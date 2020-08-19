#  TensorFlow Model inference using C API

This example demonstrates adding custom handler for TensorFlow Model inference using C API. Since custom handlers
need to be written in Python. We use [pybind11](https://github.com/pybind/pybind11) module for Python to C++ binding.


## Prerequisites

* TensorFlow C API
* A compiler with C++11 support
* CMake >= 2.8.12


## Generate Python Module

We are using [CppFlow](https://github.com/serizba/cppflow) source code, to invoke TensorFlow C API. The CppFlow
uses TensorFlow C API to run the models. Download it from [here](https://www.tensorflow.org/install/lang_c).
You can install the library system wide or place it in 'libtensorflow' in home directory.
The [CMakeList.txt](bindings/CMakeList.txt) assumes 'libtensorflow' is in home directory.

## Steps to create a custom handler
1. Create pybind module
Use commands below to create a pybind python module which can be imported in python program.
This will create a python module called "tf_c_inference".

```bash
git clone --recursive https://github.com/awslabs/multi-model-server.git

# If you haven't used recursive option while cloning, use below command to get updated submodule
git submodule update --init

pip install ./bindings
```
2. Create a python handler
Create a python custom handler which invokes 'load_model' and 'run_model' API of "tf_c_inference".
Here is the example [handler](handler.py).

3. Create a MAR file
Use model-archiver utility to create a MAR file using handler.py.
If you want to ship the model to different machine. Either ensure that you install "tf_c_inference" using step 1 or ship the .so file to that machine and ensure it's in python path.command

## Details about pybind code:
The binding code is specified in [tf_c_inference.cpp](bindings/tf_c_inference.cpp) file.
The APIs exposed are 'load_model' and 'run_model'. The singleton is used so that model is loaded only once.
The API invokes CppFlow C++ API which is a wrapper over TensorFlow C API. Feel free to modify the code as per your needs.

## Reference Links
1. [pybind11](https://github.com/pybind/pybind11),
2. [cmake_example](https://github.com/pybind/cmake_example)
3. [CppFlow](https://github.com/serizba/cppflow)
4. [Tensorflow C API](https://www.tensorflow.org/install/lang_c)