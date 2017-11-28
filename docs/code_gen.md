# API Client Code Generation

## Using MMS to Generate a Client API

MMS uses the [Swagger Code Generator](https://github.com/swagger-api/swagger-codegen) to create client APIs in any of its 40+ supported languages.

A simple example use is to use the `--api-gen` argument with `javascript` to get it to produce a JavaScript API for MMS as follows:

```bash
mxnet-model-server --gen-api javascript --models squeezenet=squeezenet_v1.1.model
```
This will create a `\build` folder with your new client API inside.

**Tip**: ownership of the build folder is required, so if you installed MMS in the context of root, you make get a permissions error. To solve this, just have your current account assume ownership of the build folder, or delete the build folder and try again.

## Exploring the Generated API

One of the first things to note in your new build folder is the `openapi.json` file. This is a faithful reproduction of the API description for the model when you run it with MMS. You can view this output on the [REST API page](rest_api.md) under API Description.

The build folder will have a README.md, so take a look at that for instructions on how to build and install prerequisites and get started with the client API.

The next thing to check out is the `\build\docs` folder and look at the `DefaultApi.md` file. A sampling of this generated file is displayed below. You can see that is providing stubbed out documentation for your new client API along with some example JavaScript for calling the model's endpoints.


# Example JavaScript Client API Output

## ModelServingApis.DefaultApi

All URIs are relative to *http://127.0.0.1:8080*

Method | HTTP request | Description
------------- | ------------- | -------------
[**apiDescription**](DefaultApi.md#apiDescription) | **GET** /api-description |
[**ping**](DefaultApi.md#ping) | **GET** /ping |
[**squeezenetPredict**](DefaultApi.md#squeezenetPredict) | **POST** /squeezenet/predict |


<a name="apiDescription"></a>
# **apiDescription**
> InlineResponse200 apiDescription()



### Example
```javascript
var ModelServingApis = require('model_serving_apis');

var apiInstance = new ModelServingApis.DefaultApi();

var callback = function(error, data, response) {
  if (error) {
    console.error(error);
  } else {
    console.log('API called successfully. Returned data: ' + data);
  }
};
apiInstance.apiDescription(callback);
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**InlineResponse200**](InlineResponse200.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

<a name="ping"></a>
# **ping**
> InlineResponse2001 ping()



### Example
```javascript
var ModelServingApis = require('model_serving_apis');

var apiInstance = new ModelServingApis.DefaultApi();

var callback = function(error, data, response) {
  if (error) {
    console.error(error);
  } else {
    console.log('API called successfully. Returned data: ' + data);
  }
};
apiInstance.ping(callback);
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**InlineResponse2001**](InlineResponse2001.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

<a name="squeezenetPredict"></a>
# **squeezenetPredict**
> InlineResponse2002 squeezenetPredict(data)



### Example
```javascript
var ModelServingApis = require('model_serving_apis');

var apiInstance = new ModelServingApis.DefaultApi();

var data = "/path/to/file.txt"; // File | data should be image which will be resized to: [3, 224, 224]


var callback = function(error, data, response) {
  if (error) {
    console.error(error);
  } else {
    console.log('API called successfully. Returned data: ' + data);
  }
};
apiInstance.squeezenetPredict(data, callback);
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **data** | **File**| data should be image which will be resized to: [3, 224, 224] |

### Return type

[**InlineResponse2002**](InlineResponse2002.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: application/json
