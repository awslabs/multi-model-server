# DMS REST API

TODO: api overview

## Endpoints

After local server is up, there will be three built-in endpoints:

1. [POST] &nbsp; host:port/\<model-name>/predict
2. [GET] &nbsp; &nbsp; host:port/ping
3. [GET] &nbsp; &nbsp; host:port/api-description


## Prediction

**curl Example**

Using `curl` is a great way to test REST APIs, but you're welcome to use your preferred tools. Just follow the pattern described here. If you skipped over it, we've already gone through a simple prediction example where we curled a picture of kitten like so:

```bash
curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "input0=@kitten.jpg"
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

TODO: Talk about how "API description" is actually an OpenAPI format

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
