# Sequence to Sequence inference with LSTM network trained on PenTreeBank data set

In this example, we show how to create a service which generates sentences with a pre-trained LSTM model with deep model server. This model is trained on [PenTreeBank data](https://catalog.ldc.upenn.edu/ldc99t42) and training detail can be found in [MXNet example](https://github.com/apache/incubator-mxnet/tree/master/example/rnn).

This model uses [MXNet Bucketing Module](https://mxnet.incubator.apache.org/how_to/bucketing.html) to deal with variable length input sentences and generates output sentences with the same length as inputs.

# Step by step to create service

## Step 1 - Download the pre-trained LSTM model files, signature file and vocabulary dictionary file.

```bash
mkdir lstm-model
wget https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb-symbol.json -P lstm-model
wget https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb-0100.params -P lstm-model
wget https://s3.amazonaws.com/model-server/models/lstm_ptb/signature.json -P lstm-model
wget https://s3.amazonaws.com/model-server/models/lstm_ptb/vocab_dict.txt -P lstm-model
wget https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb_service.py
```

## Step 2 - Check signature file

Let's take a look at signature file:
```json
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [1, 60]
    }
  ],
  "input_type": "application/json",
  "outputs": [
    {
      "data_name": "softmax",
      "data_shape": [1, 10000]
    }
  ],
  "output_type": "application/json"
}
```
Both input and output are of type application/json. Input data shape is (1, 60). For sequence to sequence models, the inputs can be variable length sequences. In the signature file the input shape should be set to the maximum length of the input sequence, which is the default bucket key. The bucket sizes are defined when training the model. In this example valid bucket sizes are 10, 20, 30, 40, 50 and 60. Default bucket key is the maximum value which is 60. Check [training details](https://github.com/apache/incubator-mxnet/blob/master/example/rnn/cudnn_lstm_bucketing.py) if you want to know more about the bucketing module in MXNet. Output shape is (1, 10000), since PTB data set contains 10,000 vocabularies.

## Step 3 - Check vocabulary dictionary file

[vocab_dict.txt](https://s3.amazonaws.com/model-server/models/lstm_ptb/vocab_dict.txt) is to store word to integer indexing information. In this example, each line in the text file represents a (word, index) pair. This file can be in different format and requires different customized parsing methods respectively.

## Step 4 - Create custom service class

Loading a NLP model in MXNet is a bit more complicated than vision models. We need to override `__init__`, `_preprocess`, `_inference` and `_postprocess` methods in a custom service class. Implementation details are in [lstm_ptb_service.py](lstm_ptb_service.py).

## Step 5 - Export model files with mxnet-model-export CLI tool

With model files together with signature and vocab_dict files in lstm-model folder, we are ready to export them to MMS model file.

```bash
mxnet-model-export --model-path lstm-model/ --model-name lstm_ptb --service-file-path lstm_ptb_service.py
```

## Step 6 - Establish inference service

lstm_ptb.model file is created by exporting model files. We also provided custom service script lstm_ptb_service.py. We are ready to establish the LSTM inference service:

```bash
mxnet-model-server --models lstm_ptb=lstm_ptb.model
```
You will see the following outputs which means the service is successfully established:

```bash
I1102 11:25:58 4873 /Users/user/anaconda/lib/python2.7/site-packages/mxnet_model_server-0.1.1-py2.7.egg/mms/mxnet_model_server.py:__init__:75] Initialized model serving.
I1102 11:25:59 4873 /Users/user/anaconda/lib/python2.7/site-packages/mxnet_model_server-0.1.1-py2.7.egg/mms/serving_frontend.py:add_endpoint:177] Adding endpoint: lstm_ptb_predict to Flask
I1102 11:25:59 4873 /Users/user/anaconda/lib/python2.7/site-packages/mxnet_model_server-0.1.1-py2.7.egg/mms/serving_frontend.py:add_endpoint:177] Adding endpoint: ping to Flask
I1102 11:25:59 4873 /Users/user/anaconda/lib/python2.7/site-packages/mxnet_model_server-0.1.1-py2.7.egg/mms/serving_frontend.py:add_endpoint:177] Adding endpoint: api-description to Flask
I1102 11:25:59 4873 /Users/user/anaconda/lib/python2.7/site-packages/mxnet_model_server-0.1.1-py2.7.egg/mms/mxnet_model_server.py:start_model_serving:88] Service started at 127.0.0.1:8080
```

The endpoint is on localhost and port 8080. You can change them by passing --host and --port when establishing the service.

## Test inference service

Now we can send post requests to the endpoint we just established.

Since the entire range of vocabularies in the training set is only 10,000, you may not get very good results with arbitrary test sentences. Instead, we recommend that you test with sentences from the [PTB test data set](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.test.txt). That being said, if you try some random text you should know that any word that isn't in that 10k vocabulary is encoded with an "invalid label" of 0. This will create a prediction result of '\n'. Note that in PTB data set, person name is represented by `<unk>`.

The key value of application/json input is 'input_sentence'. This can be a different value and preprocess method in lstm_ptb_service.py needs to be modified respectively. 

```bash
curl -X POST http://127.0.0.1:8080/lstm_ptb/predict -F "data=[{'input_sentence': 'on the exchange floor as soon as ual stopped trading we <unk> for a panic said one top floor trader'}]"
```

Prediction result will be:

```bash
{
  "prediction": "the <unk> 's the the as the 's the the 're to a <unk> <unk> <unk> analyst company trading at "
}
```

Let's try another sentence:

```bash
curl -X POST http://127.0.0.1:8080/lstm_ptb/predict -F "data=[{'input_sentence': 'while friday \'s debacle involved mainly professional traders rather than investors it left the market vulnerable to continued selling this morning traders said '}]"
```

Prediction result will be:

```bash
{
  "prediction": "the 's stock were <unk> in <unk> say than <unk> were will to <unk> to to the <unk> the week \n \n \n \n \n \n \n \n \n \n "
}
```

References
1. [How to use MXNet bucketing module](https://mxnet.incubator.apache.org/how_to/bucketing.html)
2. [LSTM trained with PennTreeBank data set](https://github.com/apache/incubator-mxnet/tree/master/example/rnn)

