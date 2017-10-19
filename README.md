## Usage:

### Installation for python2 and python3
```python
pip install deep-model-server
```

### Start serving
```python
deep-model-server --models resnet-18=https://s3.amazonaws.com/mms-models/resnet-18.model [--service mxnet_vision_service] [--gen-api python] [--port 8080] [--host 127.0.0.1]
```
#### Arguments:
1. models: required, <model_name>=<model_path> pairs. 
    (1) Model path can be a local file path or URI (s3 link, or http link).
        local file path: path/to/local/model/file or file://root/path/to/model/file
        s3 link: s3://S3_endpoint[:port]/...
        http link: http://hostname/path/to/resource

    (2) Currently, the model file has .model extension, it is actually a zip file with a .model extension packing pretrained MXNet models and model signature files. The details will be explained in **Export existing model** section

    (3) Multiple models loading are also supported by specifying multiple name path pairs
2. service: optional, our system will load input service module and will initialize mxnet models with the service defined in the module. The module should contain a valid class extends our base model service with customized preprocess and postprocess.
3. port: optional, default is 8080
4. host: optional, default is 127.0.0.1
5. gen-api: optional, this will generate an open-api formated client sdk in build folder.
6. log-file: optional, log file name. By default it is "dms_app.log".
7. log-rotation-time: optional, log rotation time. By default it is "1 H", which means one hour. Valid format is "interval when". For weekday and midnight, only "when" is required. Check https://docs.python.org/2/library/logging.handlers.html#logging.handlers.TimedRotatingFileHandler for detail values.
8. log-level: optional, log level. By default it is INFO. Possible values are NOTEST, DEBUG, INFO, ERROR AND CRITICAL. Check https://docs.python.org/2/library/logging.html#logging-levels

### Export existing model to serving model format
```python
deep-model-export --model-name resnet-18 --model-path models/resnet-18
```

#### Arguments:
1. model-name: required, prefix of exported model archive file.
2. model-path: required, directory which contains files to be packed into exported archive.

   signature.json is required to be in this directory.

   Currently 4 entries are required: 

   (1) input, which contains MXNet model input names and input shapes. It is a list contains { data_name : name, data_shape : shape } maps. Client side inputs should have the same order with the input order defined here.

   (2) input_type, which defines the MIME content type for client side inputs. Currently all inputs must have the same content type and only two MIME types, "image/jpeg" and "application/json", are supported. 

   (3) output. Similar to input, it contains MXNet model output names and output shapes. 

   (4) output_type. Similar to input_type,  currently all outputs must have the same content type and only two MIME types, "image/jpeg" and "application/json", are supported. 
   
   If model has two inputs. One is image with 3 color channels and size 28 by 28. Another is image with 3 color channels and size 32 by 32. Assume the data name of inputs in mxnet module is 'image1' and 'image2', and output is named 'softmax' with length 1000, signature.json would look like the following:
   ```
      {
        'input' : [
            { 
                'data_name' : 'image1',
                'data_shape' : [ 0, 3, 28, 28 ]
             },
             { 
                'data_name' : 'image2',
                'data_shape' : [ 0, 3, 32, 32 ]
             }
        ],
        'input_type' : 'image/jpeg',
        'output' : [ 
            {        
                'data_name' : 'softmax',
                'data_shape' : [ 0, 100 ]
            }
        ],
        'output_type' : 'application/json'
      }
   ```
   Data shape is a list of integer. It should contains batch size as the first dimension to follow MXNet data shape rule. Also 0 is a placeholder for MXNet shape and means any value is valid. Batch size should be set as 0.

   If `synset.txt` is inclued in exported archive file and each line represents a category, `MXNetBaseModel` will load this file and create `labels` attribute automatically. If this file is named differently or has a different format, you need to override `__init__' method and manually load it.

### Directly export model after training in MXNet

Another method to export model is to use `export_serving` function while completing training:
```python
   import mxnet as mx
   from mms.export_model import export_serving

   mod = mx.mod.Module(...)
   # Training process
   ...

   # Export model
   signature = { "input_type": "image/jpeg", "output_type": "application/json" }
   export_serving(mod, 'resnet-18', signature, aux_files=['synset.txt'])
```

## Endpoints:
After local server is up, there will be three built-in endpoints:
1. [POST] &nbsp; host:port/\<model-name>/predict       
2. [GET] &nbsp; &nbsp; host:port/ping                        
3. [GET] &nbsp; &nbsp; host:port/api-description             


## Prediction endpoint example:

### 1.Use curl:
Pick a local image and use the following command to send request to endpoint.
white-sleeping-kitten.jpg is a local kitten image. You can replace it with other local images.
```
curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "input0=@white-sleeping-kitten.jpg"
```
```
{
  "prediction": [
    [
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.3166358768939972
      },
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.3160117268562317
      },
      {
        "class": "n04074963 remote control, remote",
        "probability": 0.047916918992996216
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.036371976137161255
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.03163142874836922
      }
    ]
  ]
}
```
### 2.Use generated client code:
  ```python
  import swagger_client
  print swagger_client.DefaultApi().resnet18_predict('white-sleeping-kitten.jpg')
  ```
  ```
  {
    'prediction': 
      "[[{u'class': u'n02123045 tabby, tabby cat', u'probability': 0.3166358768939972}, {u'class': u'n02124075 Egyptian cat', u'probability': 0.3160117268562317}, {u'class': u'n04074963 remote control, remote', u'probability': 0.047916918992996216}, {u'class': u'n02123159 tiger cat', u'probability': 0.036371976137161255}, {u'class': u'n02127052 lynx, catamount', u'probability': 0.03163142874836922}]]"
  }
  ```
  
### Ping endpoint example:
Since ping is a GET endpoint, we can see it in a browser by visiting:

curl -X GET http://127.0.0.1:8080/ping

```
{
  "health": "healthy!"
}
```

### API description example:
This endpoint will list all the apis in OpenAPI compatible format:

curl -X GET http://127.0.0.1:8080/api-description

```
{
  "description": {
    "host": "127.0.0.1:8080", 
    "info": {
      "title": "Model Serving Apis", 
      "version": "1.0.0"
    }, 
    "paths": {
      "/api-description": {
        "get": {
          "operationId": "apiDescription", 
          "produces": [
            "application/json"
          ], 
          "responses": {
            "200": {
              "description": "OK", 
              "schema": {
                "properties": {
                  "description": {
                    "type": "string"
                  }
                }, 
                "type": "object"
              }
            }
          }
        }
      }, 
      "/ping": {
        "get": {
          "operationId": "ping", 
          "produces": [
            "application/json"
          ], 
          "responses": {
            "200": {
              "description": "OK", 
              "schema": {
                "properties": {
                  "health": {
                    "type": "string"
                  }
                }, 
                "type": "object"
              }
            }
          }
        }
      }, 
      "/resnet-18/predict": {
        "post": {
          "consumes": [
            "multipart/form-data"
          ], 
          "operationId": "resnet-18_predict", 
          "parameters": [
            {
              "description": "input0 should be image with shape: [3, 224, 224]", 
              "in": "formData", 
              "name": "input0", 
              "required": "true", 
              "type": "file"
            }
          ], 
          "produces": [
            "application/json"
          ], 
          "responses": {
            "200": {
              "description": "OK", 
              "schema": {
                "properties": {
                  "prediction": {
                    "type": "string"
                  }
                }, 
                "type": "object"
              }
            }
          }
        }
      }
    }, 
    "schemes": [
      "http"
    ], 
    "swagger": "2.0"
  }
}
```

## Multi model setup:
```python
deep-model-server --models resnet-18=file://models/resnet-18 vgg16=file://models/vgg16
```
This will setup a local host serving resnet-18 model and vgg16 model on the same port.

## Define custom service:
By passing `service` argument, you can specify your own custom service. All customized service class should be inherited from MXNetBaseService:
```python
   class MXNetBaseService(SingleNodeService):
      def __init__(self, path, synset=None, ctx=mx.cpu())
    
      def _inference(self, data)

      def _preprocess(self, data)

      def _postprocess(self, data, method='predict')
```
Usually you would like to override _preprocess and _postprocess since they are binded with specific domain of applications. We provide some [utility functions](https://github.com/deep-learning-tools/mxnet-model-server/tree/master/mms/utils) for vision and NLP applications to help user easily build basic preprocess functions.
The following example is for resnet-18 service. In this example, we don't need to change __init__ or _inference methods, which means we just need override _preprocess and _postprocess. In preprocess, we first convert image to NDArray. Then resize to 224 x 224. In post process, we return top 5 categories:
```python
   import mxnet as mx
   from mms.mxnet_utils import image
   from mms.mxnet_model_service import MXNetBaseService

   class Resnet18Service(MXNetBaseService):
       def _preprocess(self, data):
           img_arr = image.read(data)
           img_arr = image.resize(img_arr, 224, 224)
           return img_arr

       def _postprocess(self, data):
           output = data[0]
           sorted = mx.nd.argsort(output[0], is_ascend=False)
           for i in sorted[0:5]:
               response[output[0][i]] = self.labels[i]
           return response
```

## Dependencies:
Flask, MXNet, numpy, JAVA(7+, required by swagger codegen)

## Deployments:
### Docker:
We have provided docker image for mxnet cpu.
Nginx and all other dependencies are installed.
The basic usage can be found [here](docker/README.md)

## Design:
To be updated

## Testing:
python -m pytest mms/tests/unit_tests/
