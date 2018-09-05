#TODO: Update this description as and when the model export tool is being written

# Exporting Examples

## Contents of this Document
* [Export Example](#export-example)
* [Export Example with Specified Custom Service](#export-example-with-specified-custom-service)
* [Exporting with Labels and Other Assets](#exporting-with-labels-and-other-assets)
* [Export Example with Customizations](#export-example-with-customizations)

## Other Relevant Documents
* [Export Overview](export.md)
* [Export an ONNX Model](export_from_onnx.md)
* [Exported MMS Model File Tour](export_model_file_tour.md)


## Export Example

We will walk through a very simple export example using `curl` to download the necessary artifacts, then `mxnet-model-export` to generate a `.model` file that can be used with MMS.

To try this out using a SqueezeNet model checkpoint from MXNet, open your terminal and run the following:
```bash
mkdir squeezenet && cd squeezenet
curl -O https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1-0000.params
curl -O https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1-symbol.json
curl -O https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/signature.json
curl -O https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/synset.txt
```

The export tool is going to look for the following at a minimum:
* symbol file (_name_-symbol.json) - in our example it will be: `squeezenet_v1.1-symbol.json`
* params file (_name_-_checkpoint#_.params) - in our example it will be: `squeezenet_v1.1-0000.params`
* signature file (signature.json)
* labels file (synset.txt)

It's easy to export just with these artifacts to get a .model file.

We're going to tell it our model's name is `squeezenet_v1.1` with the `--model-name` argument. The name works like a prefix, so it will assume that you've named the symbol file and the params file according to the pattern _name_-symbol.json and _name_-0000.params. The "0000" can be another checkpoint if that's what you have. Then we're giving it the `--model-path` to the model's assets, which are in the current working directory, so we'll use `.` for the path.

Now that you understand what the command is going to do, run the following:

```bash
mxnet-model-export --model-name squeezenet_v1.1 --model-path .
```

This will output `squeezenet_v1.1.model` in the current working directory. Try serving it with:

```bash
mxnet-model-server --models squeezenet=squeezenet_v1.1.model
```

Try inference with:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "data=@kitten.jpg"
```

**FAQ**: What happened under the hood when the model was exported?

It's going to look at the signature file and see that you're using `"input_type": "image/jpeg"`, and assume that you want the default vision service, so it will include `mxnet_vision_service.py` for you. It will also generate a manifest file. These all get rolled up inside the MMS model file.

Would you like to know more? --> *Take a [model file tour](export_model_file_tour.md)*.

## Export Example with Specified Custom Service

Let's try the export again, but this time we will also pass the `--service` argument. So that it is very clear what is happening with the export process, we'll copy the service file to a different name and specify it when we call the export tool.

```bash
cp mxnet_vision_service.py my_awesome_service.py
mxnet-model-export --model-name squeezenet_v1.1 --model-path . --service my_awesome_service.py
```

Once the model file is exported, unzip it and take a look. Your custom service should be in there. The other service file is in there too! This means you can pack up an entire application, and specify the custom service entry point with the `--service` argument. You can also verify the `manifest.json` and see that it setup the model based on the provided files and your service specification. Here's an example extract from the manifest:

```
"Model": {
    "Description": "squeezenet",
    "Service": my_awesome_service.py",
    "Symbol": "squeezenet_v1.1-symbol.json",
    "Parameters": "squeezenet_v1.1-0000.params",
    "Signature": "signature.json",
    "Model-Name": "squeezenet_v1.1",
    "Model-Format": "MXNet-Symbolic"
```

## Exporting with Labels and Other Assets

You may have noticed that we required the `synset.txt` file in this example, but didn't mention it as part of the process. It's not even in the manifest. This is because it is required by the service file or the upstream classes that the service file is extending. In our example here it uses those labels in the post-processing step to provide human-readable inference results.

If you're curious you can look in the service file and note the line that "sort of" mentions it by require a labels file, and then if you look upstream at the [mxnet_model_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_model_service.py) you will see it specifically mentioned:
```
archive_synset = os.path.join(model_dir, 'synset.txt')
```
Of course, if you [write your own custom service](custom_service.md), you can handle labels in another way. You may want to take look at the examples too. One is for an LSTM which uses a [vocabulary labels file](https://s3.amazonaws.com/model-server/models/lstm_ptb/vocab_dict.txt), or the SSD example which uses a [much shorter synset.txt](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/synset.txt) than our SqueezeNet examples for the short list of objects it is intended to identify.

**Note**: You may get an error about a missing synset if you use a custom service that is `expecting one and you didn't provide one in the folder with the other artifacts. Each sysnet correlates to the model, so make sure you have one in the directory with your other artifacts when you try to export your model.

## Export Example with Customizations

To give you an idea of how you might download another's model, modify it, then serve it, let's try out a simple use case. The example we have been using will serve the SqueezeNet model, and upon inference requests it will return the top 5 results. Let's change the **custom service** so that it returns 10 results instead.

Open the `my_awesome_service.py` file in your text editor.

Find the function for `_postprocess` and the line that says the following:

```python
return [ndarray.top_probability(d, self.labels, top=5) for d in data]
```

Change the `top=5` to `top=10`, then save the file.

Run the export process again:
```bash
mxnet-model-export --model-name squeezenet_v1.1 --model-path . --service my_awesome_service.py
```

Run the server on the updated model:
```bash
mxnet-model-server --models squeezenet=squeezenet_v1.1.model
```

Then in a different terminal window, upload an image file the API, and see your results.
```bash
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "data=@kitten.jpg"
```

Instead of the top 5 results, you will now get the top 10!

This is just one example of customization. There are many variations, but here are a couple of ideas to get your creative juices flowing:

* You might decide that you want to take a model, grab the params as a checkpoint and retrain it using additional training images. This is often called fine tuning a model. A fine tuning tutorial using MXNet with Gluon can be found in [The Straight Dope's computer vision section](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter08_computer-vision/fine-tuning.ipynb).

* You also might decide that you want to change the labels - maybe by simplifying the results without having to retrain the entire model. A result like `"class": "n02123045 tabby, tabby cat",` could just be `cat`. You would go through the `synset.txt` file, make your edits and then export the model.
