# Management API

MMS provides a set of API allow user to manage models at runtime:
1. [Register a model](#register-a-model)
2. [Increase/decrease number of workers for specific model](#scale-workers)
3. [Describe a model's status](#describe-model)
4. [Unregister a model](#unregister-a-model)
5. [List registered models](#list-models)

Management API is listening on port 8081 and only accessible from localhost by default. To change the default setting, see [MMS Configuration](configuration.md).

Similar as [Inference API](inference_api.md), Management API also provide a [API description](#api-description) to describe management APIs with OpenAPI 3.0 specification.

## Management APIs

### Register a model

* POST /models
** model_url - Model archive download url. support:
*** local model archive (.mar), the file must be directly in model_store folder.
*** local model directory, the directory must be directly in model_store foler. This option can avoid MMS extracting .mar file to temporary folder, which will improve load time and reduce disk space usage.
*** HTTP(s) protocol. MMS can download .mar files from internet.
** model_name - Name of the model. This name will be used as {model_name} in other API as path. If this parameter is not present, modelName in MANIFEST.json will be used.
** handler - Inference handler entry-point. This value will override handler in MANIFEST.json if present.
** runtime - Runtime for the model custom service code. This value will override runtime in MANIFEST.json if present. Default PYTHON.
** batch_size - Inference batch size, default: 1.
** max_batch_delay - Maximum delay for batch aggregation, default: 100 millisecnonds.
** initial_worker - Number of initial workers to create, default: 0.",
** synchronous - Decides whether creation of worker synchronous or not, default: false.

```bash
curl -X POST "http://localhost:8081/models?url=https%3A%2F%2Fs3.amazonaws.com%2Fmodel-server%2Fmodels%2Fsqueezenet_v1.1%2Fsqueezenet_v1.1.model"

{
  "status": "Model \"squeezenet_v1.1\" registered"
}
```

User may want to create workers while register, creating initial workers may take some time, user can choose between synchronous or synchronous call to make sure initial workers are created properly.

The asynchronous call will return before trying to create workers with HTTP code 202:

```bash
curl -v -X POST "http://localhost:8081/models?url=https%3A%2F%2Fs3.amazonaws.com%2Fmodel-server%2Fmodels%2Fsqueezenet_v1.1%2Fsqueezenet_v1.1.model&initial_workers=1&synchronous=true"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 29cde8a4-898e-48df-afef-f1a827a3cbc2
< content-length: 33
< connection: keep-alive
< 
{
  "status": "Worker updated"
}
```

The synchronous call will return after all workers has be adjusted with HTTP code 200.

```bash
curl -v -X POST "http://localhost:8081/models?url=https%3A%2F%2Fs3.amazonaws.com%2Fmodel-server%2Fmodels%2Fsqueezenet_v1.1%2Fsqueezenet_v1.1.model&initial_workers=1&synchronous=true"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: c4b2804e-42b1-4d6f-9e8f-1e8901fc2c6c
< content-length: 32
< connection: keep-alive
< 
{
  "status": "Worker scaled"
}
```


### Scale workers

* PUT /models/{model_name}
** min_worker - optional minimum number of worker processes. MMS will trying maintain min_worker of workers for specified model, default 1.
** max_worker - optional maximum number of worker processes. MMS will make no more than max_worker of workers for specified model, default to min_worker.
** number_gpu - optional number of GPU worker processes to create, default 0. If number_pgu exceed, rest of workers will be running on CPU.
** synchronous - Decides whether the call is synchronous or not, default: false.
** timeout - Waiting up to the specified wait time if necessary for a worker to complete all pending requests. Use 0 to terminate backend worker process immediately. Use -1 for wait infinitely.", default -1. **Note**, not implemented yet.

User can use scale worker API to dynamically adjust number of workers to better serve different inference request load.

There are two different flavour of this API, synchronous vs asynchronous.

The asynchronous call will return immediately with HTTP code 202:

```bash
curl -v -X PUT "http://localhost:8081/models/noop?min_worker=3"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 74b65aab-dea8-470c-bb7a-5a186c7ddee6
< content-length: 33
< connection: keep-alive
< 
{
  "status": "Worker updated"
}
```

The synchronous call will return after all workers has be adjusted with HTTP code 200.

```bash
curl -v -X PUT "http://localhost:8081/models/noop?min_worker=3&synchronous=true"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: c4b2804e-42b1-4d6f-9e8f-1e8901fc2c6c
< content-length: 32
< connection: keep-alive
< 
{
  "status": "Worker scaled"
}
```

### Describe modle

* GET /models/{model_name}

User can use describe model API to get detail runtime status of a model:

```bash
curl http://localhost:8081/models/noop

{
  "modelName": "noop",
  "modelVersion": "snapshot",
  "modelUrl": "noop.mar",
  "engine": "MXNet",
  "runtime": "python",
  "minWorkers": 1,
  "maxWorkers": 1,
  "batchSize": 1,
  "maxBatchDelay": 100,
  "workers": [
    {
      "id": "9000",
      "startTime": "2018-10-02T13:44:53.034Z",
      "status": "READY",
      "gpu": false,
      "memoryUsage": 89247744
    }
  ]
}
```

### Unregister a model

* DELETE /models/{model_name}

User can unregister a model to free up system resources:

```bash
curl -X DELETE http://localhost:8081/models/noop

{
  "status": "Model \"noop\" unregistered"
}
```

### List models

* GET /models
** limit - optional integer query parameter to specify the maximum number of items to return. Default is 100.
** next_page_token - optional query parameter to query for next page, this value was return by earlier API call.

User can use this API to query current registered models:

```bash
curl "http://localhost:8081/models"
```

This API supports pagination:

```bash
curl "http://localhost:8081/models?limit=2&next_page_token=2"

{
  "nextPageToken": "4",
  "models": [
    {
      "modelName": "noop",
      "modelUrl": "noop-v1.0"
    },
    {
      "modelName": "noop_v0.1",
      "modelUrl": "noop-v0.1"
    }
  ]
}
```


## API Description

To view a full list of management API, you can use following command:

```bash
curl -X OPTIONS http://localhost:8081
```

The out is OpenAPI 3.0.1 json format. You use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for detail.

```json
{
  "openapi": "3.0.1",
  "info": {
    "title": "Model Management APIs",
    "description": "The Model Management server makes it easy to manage your live Model Server instance",
    "version": "1.0.0"
  },
  "paths": {
    "/": {
      "options": {
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
        }
      }
    },
    "/models": {
      "get": {
        "description": "List registered models in Model Server.",
        "operationId": "listModels",
        "parameters": [
          {
            "in": "query",
            "name": "limit",
            "description": "Use this parameter to specify the maximum number of items to return. When this value is present, Model Server does not return more than the specified number of items, but it might return fewer. This value is optional. If you include a value, it must be between 1 and 1000, inclusive. If you do not include a value, it defaults to 100.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "100"
            }
          },
          {
            "in": "query",
            "name": "next_page_token",
            "description": "The token to retrieve the next set of results. Model Server provides the token when the response from a previous call has more results than the maximum page size.",
            "required": false,
            "schema": {
              "type": "string"
            }
          },
          {
            "in": "query",
            "name": "model_name_pattern",
            "description": "A model name filter to list only matching models.",
            "required": false,
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
                  "type": "object",
                  "required": [
                    "models"
                  ],
                  "properties": {
                    "nextPageToken": {
                      "type": "string",
                      "description": "Use this parameter in a subsequent request after you receive a response with truncated results. Set it to the value of NextMarker from the truncated response you just received."
                    },
                    "models": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "required": [
                          "modelName",
                          "modelUrl"
                        ],
                        "properties": {
                          "modelName": {
                            "type": "string",
                            "description": "Name of the model."
                          },
                          "modelUrl": {
                            "type": "string",
                            "description": "URL of the model."
                          }
                        }
                      },
                      "description": "A list of registered models."
                    }
                  }
                }
              }
            }
          }
        }
      },
      "post": {
        "description": "Register a new model in Model Server.",
        "operationId": "registerModel",
        "parameters": [
          {
            "in": "query",
            "name": "model_url",
            "description": "Model archive download url, support local file or HTTP(s) protocol. For S3, consider use pre-signed url.",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "in": "query",
            "name": "model_name",
            "description": "Name of model. This value will override modelName in MANIFEST.json if present.",
            "required": false,
            "schema": {
              "type": "string"
            }
          },
          {
            "in": "query",
            "name": "handler",
            "description": "Inference handler entry-point. This value will override handler in MANIFEST.json if present.",
            "required": false,
            "schema": {
              "type": "string"
            }
          },
          {
            "in": "query",
            "name": "runtime",
            "description": "Runtime for the model custom service code. This value will override runtime in MANIFEST.json if present.",
            "required": false,
            "schema": {
              "type": "string",
              "enum": [
                "PYTHON",
                "PYTHON2",
                "PYTHON3"
              ]
            }
          },
          {
            "in": "query",
            "name": "batch_size",
            "description": "Inference batch size, default: 1.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "1"
            }
          },
          {
            "in": "query",
            "name": "max_batch_delay",
            "description": "Maximum delay for batch aggregation, default: 100.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "100"
            }
          },
          {
            "in": "query",
            "name": "initial_worker",
            "description": "Number of initial workers, default: 0.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "0"
            }
          },
          {
            "in": "query",
            "name": "synchronous",
            "description": "Decides whether creation of worker synchronous or not, default: false.",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": "false"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Model registered.",
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
                      "description": "Error type."
                    }
                  }
                }
              }
            }
          },
          "202": {
            "description": "Accepted.",
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
                      "description": "Error type."
                    }
                  }
                }
              }
            }
          },
          "400": {
            "description": "Unable to open dependent files specified in manifest file.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "code",
                    "type",
                    "message"
                  ],
                  "properties": {
                    "code": {
                      "type": "integer",
                      "description": "Error code."
                    },
                    "type": {
                      "type": "string",
                      "description": "Error type."
                    },
                    "message": {
                      "type": "string",
                      "description": "Error message."
                    }
                  }
                }
              }
            }
          },
          "404": {
            "description": "Unable to download model archive",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "code",
                    "type",
                    "message"
                  ],
                  "properties": {
                    "code": {
                      "type": "integer",
                      "description": "Error code."
                    },
                    "type": {
                      "type": "string",
                      "description": "Error type."
                    },
                    "message": {
                      "type": "string",
                      "description": "Error message."
                    }
                  }
                }
              }
            }
          },
          "503": {
            "description": "Model register failed.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "code",
                    "type",
                    "message"
                  ],
                  "properties": {
                    "code": {
                      "type": "integer",
                      "description": "Error code."
                    },
                    "type": {
                      "type": "string",
                      "description": "Error type."
                    },
                    "message": {
                      "type": "string",
                      "description": "Error message."
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/models/{model_name}": {
      "get": {
        "description": "Provides detailed information about the specified model.",
        "operationId": "describeModel",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model to describe.",
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
                  "type": "object",
                  "required": [
                    "modelName",
                    "modelVersion",
                    "modelUrl",
                    "minWorkers",
                    "maxWorkers",
                    "status",
                    "workers",
                    "metrics"
                  ],
                  "properties": {
                    "modelName": {
                      "type": "string",
                      "description": "Name of the model."
                    },
                    "modelVersion": {
                      "type": "string",
                      "description": "Version of the model."
                    },
                    "modelUrl": {
                      "type": "string",
                      "description": "URL of the model."
                    },
                    "minWorkers": {
                      "type": "integer",
                      "description": "Configured minimum number of worker."
                    },
                    "maxWorkers": {
                      "type": "integer",
                      "description": "Configured maximum number of worker."
                    },
                    "batchSize": {
                      "type": "integer",
                      "description": "Configured batch size."
                    },
                    "maxBatchDelay": {
                      "type": "integer",
                      "description": "Configured maximum batch delay in ms."
                    },
                    "status": {
                      "type": "string",
                      "description": "Overall health status of the model"
                    },
                    "workers": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "required": [
                          "id",
                          "startTime",
                          "status"
                        ],
                        "properties": {
                          "id": {
                            "type": "string",
                            "description": "Worker id"
                          },
                          "startTime": {
                            "type": "string",
                            "description": "Worker start time"
                          },
                          "gpu": {
                            "type": "boolean",
                            "description": "If running on GPU"
                          },
                          "status": {
                            "type": "string",
                            "description": "Worker status",
                            "enum": [
                              "READY",
                              "LOADING",
                              "UNLOADING"
                            ]
                          }
                        }
                      },
                      "description": "A list of active backend workers."
                    },
                    "metrics": {
                      "type": "object",
                      "required": [
                        "rejectedRequests",
                        "waitingQueueSize",
                        "requests"
                      ],
                      "properties": {
                        "rejectedRequests": {
                          "type": "integer",
                          "description": "Number requests has been rejected in last 10 minutes."
                        },
                        "waitingQueueSize": {
                          "type": "integer",
                          "description": "Number requests waiting in the queue."
                        },
                        "requests": {
                          "type": "integer",
                          "description": "Number requests processed in last 10 minutes."
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      "put": {
        "description": "Configure number of workers for a model, This is a asynchronous call by default. Caller need to call describeModel check if the model workers has been changed.",
        "operationId": "setAutoScale",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model to describe.",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "in": "query",
            "name": "min_worker",
            "description": "Minimum number of worker processes.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "1"
            }
          },
          {
            "in": "query",
            "name": "max_worker",
            "description": "Maximum number of worker processes.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "1"
            }
          },
          {
            "in": "query",
            "name": "number_gpu",
            "description": "Number of GPU worker processes to create.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "0"
            }
          },
          {
            "in": "query",
            "name": "synchronous",
            "description": "Decides whether the call is synchronous or not, default: false.",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": "false"
            }
          },
          {
            "in": "query",
            "name": "timeout",
            "description": "Waiting up to the specified wait time if necessary for a worker to complete all pending requests. Use 0 to terminate backend worker process immediately. Use -1 for wait infinitely.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "-1"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Model workers updated.",
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
                      "description": "Error type."
                    }
                  }
                }
              }
            }
          },
          "202": {
            "description": "Accepted.",
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
                      "description": "Error type."
                    }
                  }
                }
              }
            }
          },
          "404": {
            "description": "Model not found.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "code",
                    "type",
                    "message"
                  ],
                  "properties": {
                    "code": {
                      "type": "integer",
                      "description": "Error code."
                    },
                    "type": {
                      "type": "string",
                      "description": "Error type."
                    },
                    "message": {
                      "type": "string",
                      "description": "Error message."
                    }
                  }
                }
              }
            }
          },
          "503": {
            "description": "Model workers scale failed.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "code",
                    "type",
                    "message"
                  ],
                  "properties": {
                    "code": {
                      "type": "integer",
                      "description": "Error code."
                    },
                    "type": {
                      "type": "string",
                      "description": "Error type."
                    },
                    "message": {
                      "type": "string",
                      "description": "Error message."
                    }
                  }
                }
              }
            }
          }
        }
      },
      "delete": {
        "description": "Unregister a model from Model Server. This is an asynchronous call by default. Caller can call listModels to confirm if all the works has be terminated.",
        "operationId": "unregisterModel",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model to unregister.",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "in": "query",
            "name": "synchronous",
            "description": "Decides whether the call is synchronous or not, default: false.",
            "required": false,
            "schema": {
              "type": "boolean",
              "default": "false"
            }
          },
          {
            "in": "query",
            "name": "timeout",
            "description": "Waiting up to the specified wait time if necessary for a worker to complete all pending requests. Use 0 to terminate backend worker process immediately. Use -1 for wait infinitely.",
            "required": false,
            "schema": {
              "type": "integer",
              "default": "-1"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Model unregistered.",
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
                      "description": "Error type."
                    }
                  }
                }
              }
            }
          },
          "202": {
            "description": "Accepted.",
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
                      "description": "Error type."
                    }
                  }
                }
              }
            }
          },
          "404": {
            "description": "Model not found.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "required": [
                    "code",
                    "type",
                    "message"
                  ],
                  "properties": {
                    "code": {
                      "type": "integer",
                      "description": "Error code."
                    },
                    "type": {
                      "type": "string",
                      "description": "Error type."
                    },
                    "message": {
                      "type": "string",
                      "description": "Error message."
                    }
                  }
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
