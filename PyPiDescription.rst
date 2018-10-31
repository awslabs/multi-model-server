Project Description
===================

Apache MXNet Model Server (MMS) is a flexible and easy to use tool for
serving deep learning models exported from `MXNet <http://mxnet.io/>`__
or the Open Neural Network Exchange (`ONNX <http://onnx.ai/>`__).

Use the MMS Server CLI, or the pre-configured Docker images, to start a
service that sets up HTTP endpoints to handle model inference requests.

Detailed documentation and examples are provided in the `docs
folder <https://github.com/awslabs/mxnet-model-server/blob/master/docs/README.md>`__.

Prerequisites
-------------

* **java 8**: Required. MMS use java to serve HTTP requests. You must install java 8 (or later) and make sure java is on available in $PATH environment variable *before* installing MMS. If you have multiple java installed, you can use $JAVA_HOME environment vairable to control which java to use.
* **mxnet**: `mxnet` will not be installed by default with MMS 1.0 any more. You have to install it manually if you use MxNet.

For ubuntu:
::

    sudo apt-get install openjdk-8-jre-headless


For centos
::

    sudo yum install java-1.8.0-openjdk


For Mac:
::

    brew tap caskroom/versions
    brew update
    brew cask install java8


Install MxNet:
::

    pip install mxnet

MXNet offers MKL pip packages that will be much faster when running on Intel hardware.
To install mkl package for CPU:
::

    pip install mxnet-mkl

or for GPU instance:

::

    pip install mxnet-cu92mkl


Installation
------------

::

    pip install mxnet-model-server

Development
-----------

We welcome new contributors of all experience levels. For information on
how to install MMS for development, refer to the `MMS
docs <https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md>`__.

Important links
---------------

-  `Official source code
   repo <https://github.com/awslabs/mxnet-model-server>`__
-  `Download
   releases <https://pypi.org/project/mxnet-model-server/#files>`__
-  `Issue
   tracker <https://github.com/awslabs/mxnet-model-server/issues>`__

Source code
-----------

You can check the latest source code as follows:

::

    git clone https://github.com/awslabs/mxnet-model-server.git

Testing
-------

After installation, try out the MMS Quickstart for

- `Serving a Model <https://github.com/awslabs/mxnet-model-server/blob/master/README.md#serve-a-model>`__
- `Create a Model Archive <https://github.com/awslabs/mxnet-model-server/blob/master/README.md#model-archive>`__.

Help and Support
----------------

-  `Documentation <https://github.com/awslabs/mxnet-model-server/blob/master/docs/README.md>`__
-  `Forum <https://discuss.mxnet.io/latest>`__

Citation
--------

If you use MMS in a publication or project, please cite MMS:
https://github.com/awslabs/mxnet-model-server
