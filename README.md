Portico
=======

**_Note that Portico is still experimental_**

Portico is a flexible and easy to use tool for serving Deep Learning models.

Use Portico's Server CLI, or the pre-configured Docker images, to start a service that sets up HTTP endpoints to handle model inference requests.

Currently Portico supports only MXNet models, but we plan to extend Portico to support additional deep learning frameworks and welcome contributions.


## Quick Start

### Install Portico

Make sure you have Python installed, then run:

```bash
pip install portico
```

### Serve a Model

Once installed, you can get Portico's model serving up and running very quickly. We've provided an example object classification model for you to use:
```bash
portico-server --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

With the command above executed, you have Portico running on your host, listening for inference requests.

To test it out, download a [cute picture of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and name it `kitten.jpg`. Then run the following `curl` command to post an inference request with the image.

```bash
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "input0=@kitten.jpg"
```

The predictor endpoint will return a prediction response in JSON. It will look something like the following result:

```
{
  "prediction": [
    [
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.42514491081237793
      },
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.20608820021152496
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.1271171122789383
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.04275566339492798
      },
      {
        "class": "n02123597 Siamese cat, Siamese",
        "probability": 0.016593409702181816
      }
    ]
  ]
}
```

Now you've seen how easy it can be to serve a deep learning model with Portico! [Would you like to know more?](docs/server.md)


### Export a Portico Model

Portico enables you to package up all of your model artifacts into a single model archive, that you can then easily share or distribute. To export a model, follow these steps:

**1. Download a Model (if you don't have one handy)**

First you'll need to obtain a trained model, which typically consist of a set of files such as the files listed below. Go ahead and download these files into a new and empty folder:

* [squeezenet_v1.1-symbol.json](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-symbol.json) - contains the layers and overall structure of the neural network; the name, or prefix, here is "squeezenet_v1.1"
* [squeezenet_v1.1-0000.params](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-0000.params) - contains the parameters and the weights; again, the prefix is "squeezenet_v1.1"
* [signature.json](https://s3.amazonaws.com/model-server/models/model-example/signature.json) - defines the inputs and outputs that Portico is expecting to hand-off to the API
* [synset.txt](https://s3.amazonaws.com/model-server/models/model-example/synset.txt) - an *optional* list of labels (one per line)


**2. Export Your Model**

With the model files available locally, you can use the `portico-export` CLI to generate a `.model` file that can be used to serve inference with Portico.

Open your terminal and go to the folder that has the four files you just downloaded. In this next step we'll run `portico-export` and tell it our model's prefix is `squeezenet_v1.1` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets.

```bash
portico-export --model-name squeezenet_v1.1 --model-path .
```

This will output `squeezenet_v1.1.model` in the current working directory. This file is all you need to run Portico, serving inference requests for a simple image recognition API.

To learn more about exporting, check out [Portico export documentation](docs/export.md)


## Other Features

Browse over to the [Docs readme](docs/README.md) for the full index of documentation. This includes more examples, how to customize the API service, API endpoint details, and more.

For production deployments, we recommend using containers, and include pre-configured docker images for you to use. The basic usage can be found on the [Docker readme](docker/README.md).

## Contributing

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.
