# Character-level CNN Model in Gluon trained using Amazon Product Dataset

In this example, we show how to create a service which classifies a review into product type using [Character-level Convolutional Network Model (CNN) model](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) model by Yann LeCunn. This model is trained on [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/) and training detail can be found in a detailed tutorial from Thomas Delteil on [Character CNN training.](https://github.com/ThomasDelteil/CNN_NLP_MXNet).


# Step by step to create service

## Step 1 - Download the Gluon Char CNN model file, signature file, model parameter and classification labels file.

```bash
# Download the model file
$ wget https://s3.amazonaws.com/mms-char-cnn-files/gluon_crepe.py

# Download the parameters
$ wget https://s3.amazonaws.com/mms-char-cnn-files/crepe_gluon_epoch6.params

# Download the signature file
$ wget https://s3.amazonaws.com/mms-char-cnn-files/signature.json

# Download classification labels file
$ wget https://s3.amazonaws.com/mms-char-cnn-files/synset.txt
```

## Step 2 - Look at the Gluon model/service  file

For Gluon models on MMS, the models are defined, within the MMS service file, the skeletal structure of the file looks like follows.

```python
class GluonCrepe(HybridBlock):
    """
    Hybrid Block gluon Crepe model
    """
    def __init__(self, classes=7, **kwargs):
      ## Define model below
      pass

class CharacterCNNService(GluonImperativeBaseService):
    """
    Gluon Character-level Convolution Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        net = GluonCrepe()
        super(CharacterCNNService, self).__init__(model_name, model_dir, manifest,net gpu)
        # The 69 characters as specified in the paper
        self.ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
        # Map Alphabets to index
        self.ALPHABET_INDEX = {letter: index for index, letter in enumerate(self.ALPHABET)}
        # max-length in characters for one document
        self.FEATURE_LEN = 1014
        self.net.hybridize()
        # define _preprocess, _inference and _postprocess methods
```

As shown, the Gluon model derives from the basic gluon hybrid block. Gluon hybrid blocks, provide performance of a symbolic model with a imperative model. More on Gluon, hybrid blocks [here](https://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html).
The fully defined service file can be found under [gluon_crepe.py](gluon_crepe.py), we define `_preprocess`, `_inference`, `_postprocess` methods in this file.

## Step 3 - Check signature file

Let's take a look at signature file:
```json
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [1,1014]
    }
  ],
  "input_type": "application/json",
  "outputs": [
    {
      "data_name": "softmax",
      "data_shape": [0, 7]
    }
  ],
  "output_type": "application/json"
}

```
The input size is, limited to 1014, characters as mentioned in the paper. The output is of shape [0,7] as we classify the reviews into seven product categories. Both the input and output are passed on as 'application/json' based text content.

# Step 4 - Prepare synset.txt with list of class names

[synset.txt](synset.txt) is where we define list of all classes detected by the model. The pre-trained Character-level CNN model used in the example is trained to detect 7 classes including Books, CDs_and_Vinyl, Movies_and_TV and more. See synset.txt file for list of all classes.

The list of classes in synset.txt will be loaded by MMS as list of labels in inference logic.


## Step 5 - Export model files with mxnet-model-export CLI tool

With model file together with signature and  files in the model folder, we are ready to export them to MMS model file.

```bash
mxnet-model-export --model-path /path/to/mode/folder --model-name character_cnn --service-file-path /path/to/model/folder/gluon_crepe.py
```

A packaged model can be downloaded from [here.](https://s3.amazonaws.com/mms-char-cnn-files/character_cnn.model)

## Step 6 - Establish inference service

character_cnn.model file is created by exporting model files. We also defined custom service under gluon_crepe.py. We are ready to establish the Character-level CNN inference service:

```bash
mxnet-model-server --models crepe=character_cnn.model
```

The endpoint is on localhost and port 8080. You can change them by passing --host and --port when establishing the service.

## Test inference service

Now we can send post requests to the endpoint we just established.


The key values of application/json input are 'review_title', 'review'. This can be a different value or combined to a single input , to achieve this preprocess method in gluon_crepe.py needs to be modified.

Let's take up a movie, review

```bash
$ curl -X POST http://127.0.0.1:8080/crepe/predict -F "data=[{'review_title':'Inception is the best','review': 'great direction and story'}]"
```
Prediction result will be:

```json
{
  "prediction": [{"category":"Movies_and_TV"}]
}
```

Let's try another review, this time for a music album.

```bash
$ curl -X POST http://127.0.0.1:8080/crepe/predict -F "data=[{'review_title':'fantastic quality','review': 'quality sound playback'}]"
```

Prediction result will be:

```json
{
  "prediction":[{"category":"CDs_and_Vinyl"}]
}
```

References
1. [Character-level CNN](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
2. [How to train Character-level CNN on gluon](https://github.com/ThomasDelteil/CNN_NLP_MXNet)
