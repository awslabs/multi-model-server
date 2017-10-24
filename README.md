# Deep Model Server

The purpose of **Deep Model Server (DMS)** is to provide an easy way for you to host and serve trained models. For example, you have a model that was trained on millions of images and it's capable of providing predictions on 1,000 different classes (let's say 1,000 different birds for this example). You want to write an app that lets your users snap a picture of a bird and it'll tell them what kind of bird it might be. You can use Deep Model Server to run the bird model, intake images, and return a prediction.

You can also use DMS with **multiple models**, so it would be no problem to add a dog classifier, one for cats, and one for flowers. DMS isn't limited to *vision* type models either. Any kind of model that takes an input and returns a prediction is suitable for DMS. It can run a speech recognition model and a model for a chatbot, so you could have your very own virtual assistant service running from the same server.

Let's talk about what DMS is not. It isn't a managed service. You still need to run it on a host you manage. You still need to manage your input and output pipelines.

## Technical Details

Now that you have a high level view of DMS, let's get a little into the weeds. DMS takes a deep learning model and it wraps it in a REST API. Currently it is bundled with MXNet and it comes with a built-in web server that you run from command line. This command line call takes in the single or multiple MXNet models you want to serve, along with optional port and IP info. Additionally you can point it to service extensions which define pre-processing and post-processing steps. Currently, DMS comes with a default vision service which makes it easy to serve a image classification model. If you're looking to build chat bots or video understanding then you'll have some additional leg work to do with the pre-processing and post-processing steps.

TODO: Add overview of export features

### Supported Deep Learning Frameworks

As of this first release, DMS only supports MXNet. In future versions, DMS will support models from other frameworks! As an open source project, we welcome contributions from the community to build ever wider support and enhanced model serving functionality.

## Exporting a DMS Compatible Model

You can try out exporting a model in three easy steps. First things first though: you need to install DMS:

**1. Installation for Python 2 and Python 3**

```bash
pip install deep-model-server
```

**2. Download a Trained Model**

Each of these files is viewable in a text editor. Download and extract the provided zip file.

TODO: get S3 link zip file

* [model-example.zip]() - contains the assets needed to generate a DMS compatible model file

Given these files you can use the `deep-model-export` CLI to generate a `.model` file that can be used with DMS. To use your own model, take a look at the [DMS export documentation](docs/export.md) for details on saving a checkpoint or other model exporting options.

**3. Export a DMS Model**

Open your terminal and go to the folder you just extracted. Using the zip file and its directory structure can help you keep things organized. In this next step we'll run `deep-model-export` and tell it our model's prefix is `resnet-18` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets. These are all in the `models/resnet-18` folder.

```bash
deep-model-export --model-name resnet-18 --model-path models/resnet-18
```

This will output `resnet-18.model` in the current working directory. This file is all you need to run DMS for an easy image recognition API.

[Would you like to know more?](docs/export.md)

## Serving a Model

You can get DMS model serving up and running very quickly with the following three steps.

**1. Installation for Python 2 and Python 3**

```bash
pip install deep-model-server
```

**2. Serve the resnet-18 Model for Image Classification**

If you already tried the `deep-model-export` and have the `resnet-18.model` file then you're almost there! However, it's a good idea to move your exported models to another folder before serving them. `deep-model-server` will extract the model archive in the same folder. If you skipped to this section, you can still try it out with a one-liner, so stay tuned.

```bash
mv resnet-18.model exported-models # to keep things organized
deep-model-server --models resnet-18=exported-models/resnet-18.model
```

If you don't have a `resnet-18.model` file handy, no worries. You can use a URL with DMS, and we've provided one on S3 for you to use. Note that when you use this method, the `resnet-18.model` file will be downloaded to your current working directory and this file's contents will be extracted into a `resnet-18` folder.

```bash
deep-model-server --models resnet-18=https://s3.amazonaws.com/mms-models/resnet-18.model
```

TODO: move s3 buckets to dms-models

Either which way you host the model file, the contents are extracted and the model is served with the default options (localhost on port 8080). Also, if you already have run the URL route once, and have the model file locally it will use the local file instead.

You can test DMS and look at the API description by hitting the [api-description](http://127.0.0.1:8080/api-description) endpoint which is hosted at `http://127.0.0.1:8080/api-description`.

**3. Predict an Image!**

First, go download a [cute picture of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and name it `kitten.jpg`. Then run the following `curl` command to post the image to your DMS.

```bash
curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "input0=@kitten.jpg"
```

The predictor endpoint will return a prediction in JSON. It will look something like the following result:

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

Now you've seen how easy it can be to serve a deep learning model with Deep Model Server! [Would you like to know more?](docs/serve.md)

## Dependencies

Flask, MXNet, numpy, JAVA(7+, required by swagger codegen)

## Deployments

### Docker
We have provided a Docker image for an MXNet CPU build on Ubuntu. Nginx, Gunicorn, and all other dependencies are also pre-installed.
The basic usage can be found on the [Docker readme](docker/README.md).

## Design
To be updated
