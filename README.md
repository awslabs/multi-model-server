Model Server for Apache MXNet
=======

| ubuntu/python-2.7 | ubuntu/python-3.5 |
|---------|---------|
| ![Python3 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoicGZ6dXFmMU54UGxDaGsxUDhXclJLcFpHTnFMNld6cW5POVpNclc4Vm9BUWJNamZKMGdzbk1lOU92Z0VWQVZJTThsRUttOW8rUzgxZ2F0Ull1U1VkSHo0PSIsIml2UGFyYW1ldGVyU3BlYyI6IkJJaFc1QTEwRGhwUXY1dDgiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) | ![Python2 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiYVdIajEwVW9uZ3cvWkZqaHlaRGNUU2M0clE2aUVjelJranJoYTI3S1lHT3R5THJXdklzejU2UVM5NWlUTWdwaVVJalRwYi9GTnJ1aUxiRXIvTGhuQ2g0PSIsIml2UGFyYW1ldGVyU3BlYyI6IjArcHVCaFgvR1pTN1JoSG4iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) |

Model Server for Apache MXNet (MMS) is a flexible and easy to use tool for serving Deep Learning models.

Use MMS Server CLI, or the pre-configured Docker images, to start a service that sets up HTTP endpoints to handle model inference requests.

A quick overview and examples are below. Detailed documentation and examples are provided in the [docs folder](docs/README.md).


## Quick Start

### Install with pip

Make sure you have Python installed, then run:

```bash
pip install mxnet-model-server
```

If you get an error because of the missing [LibGFortran](https://gcc.gnu.org/onlinedocs/gfc-internals/LibGFortran.html) library, please use the following command to install.

Ubuntu:

```bash
apt-get install libgfortran3
```

Amazon Linux:

```bash
yum install gcc-gfortran
```

### Install from Source

Alternatively, you may install MMS from source:

```bash
git clone https://github.com/awslabs/mxnet-model-server.git && cd mxnet-model-server
sudo python setup.py install
```

### Serve a Model

Once installed, you can get MMS model serving up and running very quickly. We've provided an example object classification model for you to use:
```bash
mxnet-model-server \
  --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

With the command above executed, you have MMS running on your host, listening for inference requests.

To test it out, download a [cute picture of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and name it `kitten.jpg`. Then run the following `curl` command to post an inference request with the image. In the example below both of these steps are provided.

```bash
wget -O kitten.jpg \
  https://upload.wikimedia.org/wikipedia/commons/8/8f/Cute-kittens-12929201-1600-1200.jpg
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "data=@kitten.jpg"
```

The predictor endpoint will return a prediction response in JSON. It will look something like the following result:

```
{
  "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.9408261179924011
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.055966004729270935
      },
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.0025502564385533333
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.00034320182749070227
      },
      {
        "class": "n02123394 Persian cat",
        "probability": 0.00026897044153884053
      }
    ]
  ]
}
```
Other models can be downloaded from the [model zoo](docs/model_zoo.md), so try out some of those as well.

Now you've seen how easy it can be to serve a deep learning model with MMS! [Would you like to know more?](docs/server.md)


### Export a Model

MMS enables you to package up all of your model artifacts into a single model archive, that you can then easily share or distribute. To export a model, follow these **two** steps:

**1. Download Model Artifacts (if you don't have them handy)**

[Model-Artifacts.zip](https://s3.amazonaws.com/model-server/inputs/Model-Artifacts.zip) - 5 MB

 Then extract the zip file to see the following model artifacts:

* **Model Definition** (json file) - contains the layers and overall structure of the neural network
* **Model Params and Weights** (params file) - contains the parameters and the weights
* **Model Signature** (json file) - defines the inputs and outputs that MMS is expecting to hand-off to the API
* **assets** (text files) - auxiliary files that support model inference such as vocabularies, labels, etc. and vary depending on the model

Further details on these files, custom services, and advanced exporting features can be found on the [Exporting Models for Use with MMS](docs/export.md) page in the [docs folder](docs).

**2. Export Your Model**

With the model artifacts available locally, you can use the `mxnet-model-export` CLI to generate a `.model` file that can be used to serve an inference API with MMS.

Open your terminal and go to the folder that has the files you just downloaded.

In this next step we'll run `mxnet-model-export` and tell it our model's prefix is `squeezenet_v1.1` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets.

```bash
mxnet-model-export --model-name squeezenet_v1.1 --model-path .
```

This will output `squeezenet_v1.1.model` in the current working directory, and it assumes all of the model artifacts are also in the current working directory. Otherwise, instead of `.` you would use a path to the artifacts. This file is all you need to run MMS, serving inference requests for a simple image recognition API. Go back to the Serve a Model tutorial above and try to run this model that you just exported!

To learn more about exporting, check out [MMS export documentation](docs/export.md)


## Other Features

Browse over to the [Docs readme](docs/README.md) for the full index of documentation. This includes more examples, how to customize the API service, API endpoint details, and more.

For production deployments, we recommend using containers, and for this purpose we include pre-configured docker images for you to use. The basic usage can be found on the [Docker readme](docker/README.md).

## Contributing

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.
