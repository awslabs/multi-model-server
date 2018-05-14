Project Description
===================

Apache MXNet Model Server (MMS) is a flexible and easy to use tool for
serving deep learning models exported from `MXNet <http://mxnet.io/>`__
or the Open Neural Network Exchange (`ONNX <http://onnx.ai/>`__).

Use the MMS Server CLI, or the pre-configured Docker images, to start a
service that sets up HTTP endpoints to handle model inference requests.

Detailed documentation and examples are provided in the `docs
folder <docs/README.md>`__.

Prerequisites
-------------

If you wish to use ONNX with MMS, you will need to first install a
``protobuf`` compiler. This is not needed if you wish to serve MXNet
models.

`Instructions for installing MMS with
ONNX <https://github.com/awslabs/mxnet-model-server#install-with-pip>`__.

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

After installation, try out the MMS Quickstart for `Serving a
Model <https://github.com/awslabs/mxnet-model-server/blob/master/README.md#serve-a-model>`__
and `Exporting a
Model <https://github.com/awslabs/mxnet-model-server/blob/master/README.md#export-a-model>`__.

Help and Support
----------------

-  `Documentation <https://github.com/awslabs/mxnet-model-server/blob/master/docs/README.md>`__
-  `Forum <https://discuss.mxnet.io/latest>`__

Citation
--------

If you use MMS in a publication or project, please cite MMS:
https://github.com/awslabs/mxnet-model-server
