# Deep Model Server

The purpose of **Deep Model Server (DMS)** is to provide an easy way for you to host and serve trained models. For example, you have a model that was trained on millions of images and it's capable of providing predictions on 1,000 different classes (let's say 1,000 different birds for this example). You want to write an app that lets your users snap a picture of a bird and it'll tell them what kind of bird it might be. You can use Deep Model Server to run the bird model, intake images, and return a prediction.

You can also use DMS with **multiple models**, so it would be no problem to add a dog classifier, one for cats, and one for flowers. DMS isn't limited to *vision* type models either. Any kind of model that takes an input and returns a prediction is suitable for DMS. It can run a speech recognition model and a model for a chatbot, so you could have your very own virtual assistant service running from the same server.

Let's talk about what DMS is not. It isn't a managed service. You still need to run it on a host you manage. You still need to manage your input and output pipelines.

## Technical Details

Now that you have a high level view of DMS, let's get a little into the weeds. DMS takes a deep learning model and it wraps it in a REST API. Currently it is bundled with the MXNet framework, and it comes with a built-in web server that you run from command line. This command line call takes in the single or multiple models you want to serve, along with additional optional parameters controlling the port, host, and logging. Additionally, you can point it to service extensions which define pre-processing and post-processing steps. DMS also comes with a default vision service that makes it easy to serve an image classification model. If you're looking to build chat bots or video understanding then you'll have some additional leg work to do with the pre-processing and post-processing steps. These are covered in more detail in the [custom service](custom_service.md) documentation.

Another key feature that we'll demonstrate as a first example in the next section is DMS's export capability. It is a separate CLI that takes in network definitions in the form of a JSON file, the trained network weight in the form of a parameters file, and the description of the models' inputs and outputs in the form of a signature JSON file. It outputs a `.model` zip file that DMS's server CLI uses to serve the models.

### Supported Deep Learning Frameworks

As of this first release, DMS only supports MXNet. In future versions, DMS will support models from other frameworks! As an open source project, we welcome contributions from the community to build ever wider support and enhanced model serving functionality.

## Exporting a DMS Compatible Model

You can try out exporting a model in three easy steps. First things first though: you need to install DMS.

**1. Installation for Python 2 and Python 3**

```bash
pip install deep-model-server
```

**2. Download a Trained Model**

The files in the `model-example.zip` file are human-readable in a text editor, with the exception of the `.params` file: this file is binary, and is usually quite large. Download and extract the provided model file. It is a zip file under the hood, so if you have trouble extracting it, change the extension to .zip first and then extract it.

* [model-example.zip](https://s3.amazonaws.com/model-server/models/model-example/model-example.zip) - contains the following four files
* [squeezenet_v1.1-symbol.json](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-symbol.json) - contains the layers and overall structure of the neural network; the name, or prefix, here is "squeezenet_v1.1"
* [squeezenet_v1.1-0000.params](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-0000.params) - contains the parameters and the weights; again, the prefix is "squeezenet_v1.1"
* [signature.json](https://s3.amazonaws.com/model-server/models/model-example/signature.json) - defines the inputs and outputs that DMS is expecting to hand-off to the API
* [synset.txt](https://s3.amazonaws.com/model-server/models/model-example/synset.txt) - an *optional* list of labels (one per line)

Given these files you can use the `deep-model-export` CLI to generate a `.model` file that can be used with DMS. To use your own model, take a look at the [DMS export documentation](docs/export.md) for details on saving a checkpoint or other model exporting options.

**3. Export a DMS Model**

Open your terminal and go to the folder you just extracted. Using the zip file and its directory structure can help you keep things organized. In this next step we'll run `deep-model-export` and tell it our model's prefix is `squeezenet_v1.1` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets. These are all in the `models/squeezenet_v1.1` folder.

```bash
deep-model-export --model-name squeezenet_v1.1 --model-path models/squeezenet_v1.1
```

This will output `squeezenet_v1.1.model` in the current working directory. This file is all you need to run DMS for an easy image recognition API.

[Would you like to know more?](docs/export.md)

## Serving a Model

You can get DMS model serving up and running very quickly with the following three steps.

**1. Installation for Python 2 and Python 3**

```bash
pip install deep-model-server
```

**2. Serve the squeezenet_v1.1 Model for Image Classification**

If you already tried the `deep-model-export` and have the `squeezenet_v1.1.model` file then you're almost there! However, it's a good idea to move your exported models to another folder before serving them. `deep-model-server` will extract the model archive in the same folder. If you skipped to this section, you can still try it out with a one-liner, so stay tuned.

```bash
mv squeezenet_v1.1.model exported-models # to keep things organized
deep-model-server --models squeezenet=exported-models/squeezenet_v1.1.model
```

If you don't have a `squeezenet_v1.1.model` file handy, no worries. You can use a URL with DMS, and we've provided one on S3 for you to use. Note that when you use this method, the `squeezenet_v1.1.model` file will be downloaded to your current working directory and this file's contents will be extracted into a `squeezenet_v1.1` folder.

```bash
deep-model-server --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

Note that we used `squeezenet` to name the model in the API instead of using `squeezenet_v1.1` which is the model's prefix. When you serve a model, you can call it whatever you want. We could have a model named `super-fancy-net.model` that can detect a lot of things, but if we customized and limited DMS's inference with a custom service, we might want the "not-hot-dog" API. In this case we might invoke DMS like this:

```bash
deep-model-server --models not-hot-dog=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/super-fancy-net.model --service nothotdog.py
```

For more info on custom services take a look at [custom service docs](custom_service.md) and the [object recognition example](../examples/ssd/README.md).

Now back to how you access the model file. Either which way you host the model file (S3 or local) when calling `deep-model-server`, the contents of the .model file are extracted and the model is served with the default options (localhost on port 8080). Also, if you already have run the URL route once, and have the model file locally it will use the local file instead.

You can test DMS and look at the API description by hitting the [api-description](http://127.0.0.1:8080/api-description) endpoint which is hosted at `http://127.0.0.1:8080/api-description`. You will see the endpoint's name in there that you you used when starting up `deep-model-server`. The part you would look for in the JSON response is the predict endpoint. It will look something like this:

```
"/squeezenet/predict": {
  "post": {
    "consumes": [
      "multipart/form-data"
```

**3. Predict an Image!**

First, go download a [cute picture of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and name it `kitten.jpg`. Then run the following `curl` command to post the image to your DMS. We're going to use the API endpoint that we saw when looking at the `api-description`. In this example, `/squeezenet/predict`.

```bash
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "input0=@kitten.jpg"
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

## Other Features

Browse over to the [Docs readme](docs/README.md) for the full index of documentation. This includes more examples, how to customize the API service, API endpoint details, and more.

## Deployments

### Docker
We have provided a Docker image for an MXNet CPU build on Ubuntu. Nginx, Gunicorn, and all other dependencies are also pre-installed.
The basic usage can be found on the [Docker readme](docker/README.md).

## Design
To be updated
