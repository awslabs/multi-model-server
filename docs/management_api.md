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

`POST /models`
* model_url - Model archive download url. support:
    * local model archive (.mar), the file must be directly in model_store folder.
    * local model directory, the directory must be directly in model_store foler. This option can avoid MMS extracting .mar file to temporary folder, which will improve load time and reduce disk space usage.
    * HTTP(s) protocol. MMS can download .mar files from internet.
* model_name - Name of the model. This name will be used as {model_name} in other API as path. If this parameter is not present, modelName in MANIFEST.json will be used.
* handler - Inference handler entry-point. This value will override handler in MANIFEST.json if present.
* runtime - Runtime for the model custom service code. This value will override runtime in MANIFEST.json if present. Default PYTHON.
* batch_size - Inference batch size, default: 1.
* max_batch_delay - Maximum delay for batch aggregation, default: 100 millisecnonds.
* initial_worker - Number of initial workers to create, default: 0.",
* synchronous - Decides whether creation of worker synchronous or not, default: false.

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

`PUT /models/{model_name}`
* min_worker - optional minimum number of worker processes. MMS will trying maintain min_worker of workers for specified model, default 1.
* max_worker - optional maximum number of worker processes. MMS will make no more than max_worker of workers for specified model, default to min_worker.
* number_gpu - optional number of GPU worker processes to create, default 0. If number_pgu exceed, rest of workers will be running on CPU.
* synchronous - Decides whether the call is synchronous or not, default: false.
* timeout - Waiting up to the specified wait time if necessary for a worker to complete all pending requests. Use 0 to terminate backend worker process immediately. Use -1 for wait infinitely.", default -1. **Note**, not implemented yet.

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

`GET /models/{model_name}`

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

`DELETE /models/{model_name}`

User can unregister a model to free up system resources:

```bash
curl -X DELETE http://localhost:8081/models/noop

{
  "status": "Model \"noop\" unregistered"
}
```

### List models

`GET /models`
* limit - optional integer query parameter to specify the maximum number of items to return. Default is 100.
* next_page_token - optional query parameter to query for next page, this value was return by earlier API call.

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

`OPTIONS /`

To view a full list of inference and management API, you can use following command:

```bash
# To view all inference API:
curl -X OPTIONS http://localhost:8080

# To view all management API:
curl -X OPTIONS http://localhost:8081
```

The out is OpenAPI 3.0.1 json format. You use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for detail.

See example output:
* [Inference API description output](../frontend/server/src/test/resources/inference_open_api.json)
* [Management API description output](../frontend/server/src/test/resources/management_open_api.json)
