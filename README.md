## Usage:
### Installation
```python
pip install mms
```
### Start serving
```python
mms --models resnet-18=https://github.com/yuruofeifei/mms/raw/master/models/resnet-18.zip [--process mxnet_vision_service] [--gen-api python] [--port 8080]
```
#### Arguments:
1. models: required, model_name=model_path pairs, multiple models are supported.
2. process: optional, our system will load input module and will initialize mxnet models with the service defined in the module. The module should contain a valid class extends our base model service with customized preprocess and postprocess.
3. gen-api: optional, this will generate an open-api formated client sdk in build folder.
4. port: optional, default 8080

### Export existing model to serving model format
```python
mms_export --model resnet-18=models/resnet-18.zip --signature signature.json --synset synset.txt --export-path models
```
#### Arguments:
1. model: required, model_name=model_path pair. Model path is the  path to pre-trained model file.
2. signature: required, signature json file for model service.
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
3. synset: optional, a synset file for classification task. [Here](https://github.com/tornadomeet/ResNet/blob/master/predict/synset.txt) is the synset file for Imagenet-11k.
   The format looks like following:
   ```
      n01440764 tench, Tinca tinca

      n01443537 goldfish, Carassius auratus

      n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias

      ...
   ```

## Endpoints:
After local server is up, there will be three built-in endpoints:
1. [POST] &nbsp; host:port/\<model-name>/predict       
2. [GET] &nbsp; &nbsp; host:port/ping                        
3. [GET] &nbsp; &nbsp; host:port/api-description             


## Prediction endpoint example:

### 1.Use curl:
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

http://127.0.0.1:8080/ping

```
{
  "health": "healthy!"
}
```

### API description example:
This endpoint will list all the apis in OpenAPI compatible format:

http://127.0.0.1:8080/api-description

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
python mms.py --model resnet-18=file://models/resnet-18 vgg16=file://models/vgg16
```
This will setup a local host serving resnet-18 model and vgg16 model on the same port.

## Define your own service:
By passing `process` argument, you can specify your own service. All customized service class should be inherited from MXNetBaseService:
```python
   class MXNetBaseService(SingleNodeService):
      def __init__(self, path, synset=None, ctx=mx.cpu())
    
      def _inference(self, data)

      def _preprocess(self, data)

      def _postprocess(self, data, method='predict')
```
Usually you would like to override _preprocess and _postprocess since they are binded with specific domain of applications. We provide some [utility functions](https://github.com/yuruofeifei/mms/blob/master/utils/mxnet_utils.py) for vision and NLP applications to help user easily build basic preprocess functions.
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


## Design:
To be updated

## Testing:
python -m unittest tests/unit_tests/test_serving_frontend
