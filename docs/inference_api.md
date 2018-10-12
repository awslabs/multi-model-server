# Inference API

Inference API is listening on port 8080 and only accessible from localhost by default. To change the default setting, see [MMS Configuration](configuration.md).

There are three type of APIs:

1. [API description](#api-description) - Describe MMS inference APIs with OpenAPI 3.0 specification
2. [Health check API](#health-check-api) - Check MMS health status
3. [Predictions API](#predictions-api) - Make predictions API call to MMS

## API Description

To view a full list of inference API, you can use following command:

```bash
curl -X OPTIONS http://localhost:8443
```

The out is OpenAPI 3.0.1 json format. You use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for detail.

```json
{
  "openapi": "3.0.1",
  "info": {
    "title": "Model Serving APIs",
    "description": "Model Server is a flexible and easy to use tool for serving deep learning models",
    "version": "1.0.0"
  },
  "paths": {
    "/api-description": {
      "get": {
        "operationId": "apiDescription",
        "parameters": [],
        "responses": {
          "200": {
            "description": "A openapi 3.0.1 descriptor.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "openapi",
                    "info",
                    "paths"
                  ],
                  "properties": {
                    "openapi": {
                      "type": "string"
                    },
                    "info": {
                      "type": "object"
                    },
                    "paths": {
                      "type": "object"
                    }
                  }
                }
              }
            }
          }
        },
        "deprecated": true
      }
    },
    "/{model_name}/predict": {
      "post": {
        "description": "A legacy predict entry point for each model.",
        "operationId": "predict",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model to unregister.",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "Input data format is defined by each model.",
          "content": {
            "*/*": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Output data format is defined by each model.",
            "content": {
              "*/*": {
                "schema": {
                  "type": "string",
                  "format": "binary"
                }
              }
            }
          }
        },
        "deprecated": true
      }
    },
    "/ping": {
      "get": {
        "operationId": "ping",
        "parameters": [],
        "responses": {
          "200": {
            "description": "Model server status.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "status"
                  ],
                  "properties": {
                    "status": {
                      "type": "string",
                      "description": "Overall status of the Model Server."
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/invocations": {
      "post": {
        "description": "A generic invocation entry point for all models.",
        "operationId": "invocations",
        "parameters": [
          {
            "in": "query",
            "name": "model_name",
            "description": "Name of model",
            "required": false,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "content": {
            "multipart/form-data": {
              "schema": {
                "required": [
                  "data"
                ],
                "properties": {
                  "model_name": {
                    "type": "string",
                    "description": "Name of model"
                  },
                  "data": {
                    "type": "string",
                    "format": "binary",
                    "description": "Inference input data"
                  }
                }
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "OK"
          }
        },
        "deprecated": true
      }
    },
    "/predictions/{model_name}": {
      "post": {
        "description": "Predictions entry point for each model. Use OPTIONS method to get detailed model API input and output description.",
        "operationId": "predictions",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model.",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "Input data format is defined by each model. Use OPTIONS method to get details for model input format.",
          "content": {
            "*/*": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Output data format is defined by each model. Use OPTIONS method to get details for model output and output format.",
            "content": {
              "*/*": {
                "schema": {
                  "type": "string",
                  "format": "binary"
                }
              }
            }
          }
        }
      },
      "options": {
        "description": "Display details of per model input and output.",
        "operationId": "predictionsApi",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model.",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "OK",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

## Health check API

MMS support a `ping` API that user can check MMS health status:

```bash
curl http://localhost:8080/ping
```

Your response, if the server is running should be:

```json
{
  "health": "healthy!"
}
```

## Predictions API

MMS 1.0 support 0.4 style API calls, those APIs are deprecated, they will be removed in future release. See [Deprecated APIs](#deprecated-api) for detail.

For each loaded model, user can make REST call to URI: /predictions/{model_name}

* POST /predictions/{model_name}

**curl Example**

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg

curl -X POST http://localhost:8080/predictions/resnet-18 -T kitten.jpg

or:

curl -X POST http://localhost:8080/predictions/resnet-18 -F "data=@kitten.jpg"
```

The result was some JSON that told us our image likely held a tabby cat. The highest prediction was:

```json
{
    "class": "n02123045 tabby, tabby cat",
    "probability": 0.42514491081237793,
    ...
}
```

## Deprecated API

MMS 0.4 style predict API is kept for backward compatible purpose, and will be removed in future release.

* POST /{model_name}/predict

**curl Example**

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg

curl -X POST http://localhost:8080/resnet-18/predict -F "data=@kitten.jpg"
```
