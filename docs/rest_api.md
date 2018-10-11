# MMS REST API

MMS use RESTful API for both inference and management calls. The API is compliance with [OpenAPI specification 3.0](https://swagger.io/specification/). User can easily generate client side code for Java, Scala, C#, Javascript use [swagger codegen](https://swagger.io/swagger-codegen/).

When MMS startup, it start two web services:
* [Inference API](inference_api.md)
* [Management API](management_api.md)

By default, MMS listening on 8080 port for Inference API and 8081 on Management API.
Both API is only accessible from localhost. Please see [MMS Configuration](configuration.md) for how to enable access from remote host. 


## Inference API


After local server is up, there will be three built-in endpoints:

1. [POST] &nbsp; host:port/\<model-name>/predict
2. [GET] &nbsp; &nbsp; host:port/ping
3. [GET] &nbsp; &nbsp; host:port/api-description


## Prediction

**curl Example**

Using `curl` is a great way to test REST APIs, but you're welcome to use your preferred tools. Just follow the pattern described here. If you skipped over it, we've already gone through a simple prediction example where we curled a picture of kitten like so:

```bash
curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "data=@kitten.jpg"
```

The result was some JSON that told us our image likely held a tabby cat. The highest prediction was:

```json
"class": "n02123045 tabby, tabby cat",
"probability": 0.42514491081237793
```

## Ping

Since `ping` is a GET endpoint, we can see it in a browser by visiting:

http://127.0.0.1:8080/ping

Your response, if the server is running should be:

```json
{
  "health": "healthy!"
}
```

Otherwise, you'll probably get a server not responding error or `ERR_CONNECTION_REFUSED`.

## API Description

To view a full list of all of the end points, you want to hit `api-description`.
http://127.0.0.1:8080/api-description

Your result will be like the following, and note that if you run this on a server running multiple models all of the models' endpoints should appear here:

```json
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
              "description": "data should be image with shape: [3, 224, 224]",
              "in": "formData",
              "name": "data",
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
