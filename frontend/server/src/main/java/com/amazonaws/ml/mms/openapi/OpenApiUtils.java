/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.openapi;

import com.amazonaws.ml.mms.archive.Manifest;
import com.amazonaws.ml.mms.archive.Signature;
import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.wlm.Model;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class OpenApiUtils {

    private OpenApiUtils() {}

    public static String listApis() {
        OpenApi openApi = new OpenApi();
        Info info = new Info();
        info.setTitle("Model Serving APIs");
        info.setDescription(
                "Model Server is a flexible and easy to use tool for serving deep learning models");
        info.setVersion("1.0.0");
        openApi.setInfo(info);

        openApi.addPath("/api-description", getApiDescriptionPath());
        openApi.addPath("/{model_name}/predict", getLegacyPredictPath());
        openApi.addPath("/", getListApisPath());
        openApi.addPath("/ping", getPingPath());
        openApi.addPath("/invocations", getInvocationsPath());
        openApi.addPath("/predictions/{model_name}", getPredictionsPath());
        openApi.addPath("/models", getModelsPath());
        openApi.addPath("/models/{model_name}", getModelManagerPath());

        return JsonUtils.GSON_PRETTY.toJson(openApi);
    }

    public static String getModelApi(Model model) {
        String modelName = model.getModelName();
        OpenApi openApi = new OpenApi();
        Info info = new Info();
        info.setTitle("RESTful API for: " + modelName);
        info.setVersion("1.0.0");
        openApi.setInfo(info);

        Signature signature = model.getModelArchive().getSignature();
        openApi.addPath("/prediction/" + modelName, getModelPath(modelName, signature));

        return JsonUtils.GSON_PRETTY.toJson(openApi);
    }

    private static Path getApiDescriptionPath() {
        Schema schema = new Schema("object");
        schema.addProperty("openapi", new Schema("string"), true);
        schema.addProperty("info", new Schema("object"), true);
        schema.addProperty("paths", new Schema("object"), true);
        MediaType mediaType = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        Response resp = new Response("200", "A openapi 3.0.1 descriptor.");
        resp.addContent(mediaType);

        Operation operation = new Operation("apiDescription");
        operation.addResponse(resp);
        operation.setDeprecated(true);

        Path path = new Path();
        path.setGet(operation);
        return path;
    }

    private static Path getListApisPath() {
        Schema schema = new Schema("object");
        schema.addProperty("openapi", new Schema("string"), true);
        schema.addProperty("info", new Schema("object"), true);
        schema.addProperty("paths", new Schema("object"), true);
        MediaType mediaType = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        Response resp =
                new Response("200", "List available RESTful APIs with openapi 3.0.1 descriptor.");
        resp.addContent(mediaType);

        Operation operation = new Operation("listAPIs");
        operation.addResponse(resp);

        Path path = new Path();
        path.setOptions(operation);
        return path;
    }

    private static Path getPingPath() {
        Schema schema = new Schema("object");
        schema.addProperty(
                "status", new Schema("string", "Overall status of the Model Server."), true);
        MediaType mediaType = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        Response resp = new Response("200", "Model server status.");
        resp.addContent(mediaType);

        Operation operation = new Operation("ping");
        operation.addResponse(resp);

        Path path = new Path();
        path.setGet(operation);
        return path;
    }

    private static Path getInvocationsPath() {
        Schema schema = new Schema();
        schema.addProperty("model_name", new Schema("string", "Name of model"), false);

        Schema dataProp = new Schema("string", "Inference input data");
        dataProp.setFormat("binary");
        schema.addProperty("data", dataProp, true);

        MediaType multipart =
                new MediaType(HttpHeaderValues.MULTIPART_FORM_DATA.toString(), schema);

        RequestBody requestBody = new RequestBody();
        requestBody.setRequired(true);
        requestBody.addContent(multipart);

        Operation operation =
                new Operation("invocations", "A generic invocation entry point for all models.");
        operation.setRequestBody(requestBody);
        operation.addParameter(new QueryParameter("model_name", "Name of model"));

        Response resp = new Response("200", "OK");
        operation.addResponse(resp);
        operation.setDeprecated(true);

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getPredictionsPath() {
        Operation post =
                new Operation(
                        "predictions",
                        "Predictions entry point for each model."
                                + " Use OPTIONS method to get detailed model API input and output description.");
        post.addParameter(new PathParameter("model_name", "Name of model."));

        Schema schema = new Schema("string");
        schema.setFormat("binary");
        MediaType mediaType = new MediaType("*/*", schema);
        RequestBody requestBody = new RequestBody();
        requestBody.setDescription(
                "Input data format is defined by each model. Use OPTIONS method to get details for model input format.");
        requestBody.setRequired(true);
        requestBody.addContent(mediaType);

        post.setRequestBody(requestBody);

        schema = new Schema("string");
        schema.setFormat("binary");
        mediaType = new MediaType("*/*", schema);

        Response resp = new Response("200", "OK");
        resp.setDescription(
                "Output data format is defined by each model. Use OPTIONS method to get details for model output and output format.");
        resp.addContent(mediaType);

        post.addResponse(resp);

        Operation options =
                new Operation("predictionsApi", "Display details of per model input and output.");
        options.addParameter(new PathParameter("model_name", "Name of model."));

        resp = new Response("200", "OK");
        resp.addContent(new MediaType("application/json", new Schema("object")));
        options.addResponse(resp);

        Path path = new Path();
        path.setPost(post);
        path.setOptions(options);
        return path;
    }

    private static Path getLegacyPredictPath() {
        Operation operation =
                new Operation("predict", "A legacy predict entry point for each model.");
        operation.addParameter(new PathParameter("model_name", "Name of model to unregister."));

        Schema schema = new Schema("string");
        schema.setFormat("binary");
        MediaType mediaType = new MediaType("*/*", schema);
        RequestBody requestBody = new RequestBody();
        requestBody.setRequired(true);
        requestBody.setDescription("Input data format is defined by each model.");
        requestBody.addContent(mediaType);

        operation.setRequestBody(requestBody);

        schema = new Schema("string");
        schema.setFormat("binary");
        mediaType = new MediaType("*/*", schema);

        Response resp = new Response("200", "OK");
        resp.setDescription("Output data format is defined by each model.");
        resp.addContent(mediaType);

        operation.addResponse(resp);
        operation.setDeprecated(true);

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getModelsPath() {
        Path path = new Path();
        path.setGet(getListModelsOperation());
        path.setPost(getRegisterOperation());
        return path;
    }

    private static Path getModelManagerPath() {
        Path path = new Path();
        path.setGet(getDescribeModelOperation());
        path.setPut(getScaleOperation());
        path.setDelete(getUnRegisterOperation());
        return path;
    }

    private static Operation getListModelsOperation() {
        Operation operation =
                new Operation("listModels", "List registered models in Model Server.");

        operation.addParameter(
                new QueryParameter(
                        "limit",
                        "integer",
                        "100",
                        "Use this parameter to specify the maximum number of items to return. When"
                                + " this value is present, Model Server does not return more than the specified"
                                + " number of items, but it might return fewer. This value is optional. If you"
                                + " include a value, it must be between 1 and 1000, inclusive. If you do not"
                                + " include a value, it defaults to 100."));
        operation.addParameter(
                new QueryParameter(
                        "next_page_token",
                        "The token to retrieve the next set of results. Model Server provides the"
                                + " token when the response from a previous call has more results than the"
                                + " maximum page size."));
        operation.addParameter(
                new QueryParameter(
                        "model_name_pattern", "A model name filter to list only matching models."));

        Schema schema = new Schema("object");
        schema.addProperty(
                "nextPageToken",
                new Schema(
                        "string",
                        "Use this parameter in a subsequent request after you receive a response"
                                + " with truncated results. Set it to the value of NextMarker from the"
                                + " truncated response you just received."),
                false);

        Schema modelProp = new Schema("object");
        modelProp.addProperty("modelName", new Schema("string", "Name of the model."), true);
        modelProp.addProperty("modelUrl", new Schema("string", "URL of the model."), true);
        Schema modelsProp = new Schema("array", "A list of registered models.");
        modelsProp.setItems(modelProp);
        schema.addProperty("models", modelsProp, true);
        MediaType json = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        Response resp = new Response("200", "OK");
        resp.addContent(json);

        operation.addResponse(resp);

        return operation;
    }

    private static Operation getRegisterOperation() {
        Operation operation =
                new Operation("registerModel", "Register a new model in Model Server.");

        operation.addParameter(
                new QueryParameter(
                        "model_url",
                        "string",
                        null,
                        true,
                        "Model archive download url, support local file or HTTP(s) protocol."
                                + " For S3, consider use pre-signed url."));
        operation.addParameter(
                new QueryParameter(
                        "model_name",
                        "Name of model. This value will override modelName in MANIFEST.json if present."));
        operation.addParameter(
                new QueryParameter(
                        "handler",
                        "Inference handler entry-point. This value will override handler in MANIFEST.json if present."));

        Parameter runtime =
                new QueryParameter(
                        "runtime",
                        "Runtime for the model custom service code. This value will override runtime in MANIFEST.json if present.");
        operation.addParameter(runtime);
        operation.addParameter(
                new QueryParameter("batch_size", "Inference batch size, default: 1."));
        operation.addParameter(
                new QueryParameter(
                        "max_batch_delay", "Maximum delay for batch aggregation, default: 100."));

        Manifest.RuntimeType[] types = Manifest.RuntimeType.values();
        List<String> runtimeTypes = new ArrayList<>(types.length);
        for (Manifest.RuntimeType type : types) {
            runtimeTypes.add(type.toString());
        }
        runtime.getSchema().setEnumeration(runtimeTypes);

        operation.addResponse(new Response("200", "Register success"));
        operation.addResponse(new Response("400", "Invalid model URL."));
        operation.addResponse(new Response("400", "Missing modelName parameter."));
        operation.addResponse(new Response("400", "Missing handler parameter."));
        operation.addResponse(new Response("400", "Missing runtime parameter."));
        operation.addResponse(
                new Response("400", "Invalid model archive file format, expecting a zip file."));
        operation.addResponse(new Response("400", "Unable to parse model archive manifest file."));
        operation.addResponse(
                new Response("400", "Unable to open dependent files specified in manifest file."));
        operation.addResponse(new Response("404", "Unable to download model archive"));

        return operation;
    }

    private static Operation getUnRegisterOperation() {
        Operation operation =
                new Operation(
                        "unregisterModel",
                        "Unregister a model from Model Server. This is an asynchronized call."
                                + " Caller need to call listModels to confirm if all the works has be terminated.");

        operation.addParameter(new PathParameter("model_name", "Name of model to unregister."));
        operation.addParameter(
                new QueryParameter(
                        "timeout",
                        "integer",
                        "-1",
                        "Waiting up to the specified wait time if necessary for"
                                + " a worker to complete all pending requests. Use 0 to terminate backend"
                                + " worker process immediately. Use -1 for wait infinitely."));

        operation.addResponse(new Response("202", "Accepted."));
        operation.addResponse(new Response("404", "Model not found."));

        return operation;
    }

    private static Operation getDescribeModelOperation() {
        Operation operation =
                new Operation(
                        "describeModel",
                        "Provides detailed information about the specified model.");

        operation.addParameter(new PathParameter("model_name", "Name of model to describe."));

        Schema schema = new Schema("object");
        schema.addProperty("modelName", new Schema("string", "Name of the model."), true);
        schema.addProperty("modelVersion", new Schema("string", "Version of the model."), true);
        schema.addProperty("modelUrl", new Schema("string", "URL of the model."), true);
        schema.addProperty(
                "minWorkers", new Schema("integer", "Configured minimum number of worker."), true);
        schema.addProperty(
                "maxWorkers", new Schema("integer", "Configured maximum number of worker."), true);
        schema.addProperty("batchSize", new Schema("integer", "Configured batch size."), false);
        schema.addProperty(
                "maxBatchDelay",
                new Schema("integer", "Configured maximum batch delay in ms."),
                false);
        schema.addProperty(
                "status", new Schema("string", "Overall health status of the model"), true);

        Schema workers = new Schema("array", "A list of active backend workers.");
        Schema worker = new Schema("object");
        worker.addProperty("id", new Schema("string", "Worker id"), true);
        worker.addProperty("startTime", new Schema("string", "Worker start time"), true);
        worker.addProperty("gpu", new Schema("boolean", "If running on GPU"), false);
        Schema workerStatus = new Schema("string", "Worker status");
        List<String> status = new ArrayList<>();
        status.add("READY");
        status.add("LOADING");
        status.add("UNLOADING");
        workerStatus.setEnumeration(status);
        worker.addProperty("status", workerStatus, true);
        workers.setItems(worker);

        schema.addProperty("workers", workers, true);
        Schema metrics = new Schema("object");
        metrics.addProperty(
                "rejectedRequests",
                new Schema("integer", "Number requests has been rejected in last 10 minutes."),
                true);
        metrics.addProperty(
                "waitingQueueSize",
                new Schema("integer", "Number requests waiting in the queue."),
                true);
        metrics.addProperty(
                "requests",
                new Schema("integer", "Number requests processed in last 10 minutes."),
                true);
        schema.addProperty("metrics", metrics, true);

        MediaType mediaType = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        Response resp = new Response("200", "OK");
        resp.addContent(mediaType);
        operation.addResponse(resp);

        return operation;
    }

    private static Operation getScaleOperation() {
        Operation operation =
                new Operation(
                        "setAutoScale",
                        "Configure number of workers for a model, This is a asynchronized call."
                                + " Caller need to call describeModel check if the model workers has been changed.");
        operation.addParameter(new PathParameter("model_name", "Name of model to describe."));
        operation.addParameter(
                new QueryParameter(
                        "minWorker", "integer", "1", "Minimum number of worker processes."));
        operation.addParameter(
                new QueryParameter(
                        "maxWorker", "integer", "1", "Maximum number of worker processes."));
        operation.addParameter(
                new QueryParameter(
                        "numberGpu", "integer", "0", "Number of GPU worker processes to create."));
        operation.addParameter(
                new QueryParameter(
                        "timeout",
                        "integer",
                        "-1",
                        "Waiting up to the specified wait time if necessary for"
                                + " a worker to complete all pending requests. Use 0 to terminate backend"
                                + " worker process immediately. Use -1 for wait infinitely."));

        operation.addResponse(new Response("202", "Accepted."));
        operation.addResponse(new Response("404", "Model not found."));

        return operation;
    }

    private static Path getModelPath(String modelName, Signature signature) {
        Operation operation =
                new Operation(modelName, "A predict entry point for model: " + modelName + '.');
        if (signature == null) {
            Response resp = new Response("200", "OK");
            operation.addResponse(resp);
            Path path = new Path();
            path.setPost(operation);
            return path;
        }

        Map<String, List<Signature.Parameter>> requests = signature.getRequest();
        if (!requests.isEmpty()) {
            RequestBody body = new RequestBody();
            for (Map.Entry<String, List<Signature.Parameter>> entry : requests.entrySet()) {
                String contentType = entry.getKey();
                List<Signature.Parameter> parameters = entry.getValue();
                MediaType mediaType = getMediaType(contentType, parameters);
                body.addContent(mediaType);
            }
            operation.setRequestBody(body);
        }

        Response response = new Response("200", "OK");

        Map<String, List<Signature.Parameter>> responses = signature.getResponse();
        for (Map.Entry<String, List<Signature.Parameter>> entry : responses.entrySet()) {
            String contentType = entry.getKey();
            List<Signature.Parameter> parameters = entry.getValue();
            MediaType mediaType = getMediaType(contentType, parameters);
            response.addContent(mediaType);
        }

        operation.addResponse(response);

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static MediaType getMediaType(
            String contentType, List<Signature.Parameter> parameters) {
        CharSequence mimeType;
        if (contentType != null) {
            mimeType = HttpUtil.getMimeType(contentType);
        } else {
            mimeType = HttpHeaderValues.APPLICATION_OCTET_STREAM;
        }

        Schema schema = new Schema();
        MediaType mediaType = new MediaType(mimeType.toString(), schema);

        if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(mimeType)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        mimeType)) {
            schema.setType("object");
            for (Signature.Parameter parameter : parameters) {
                schema.addProperty(
                        parameter.getName(),
                        new Schema("string", parameter.getDescription()),
                        parameter.isRequired());
            }
        } else if (HttpHeaderValues.MULTIPART_FORM_DATA.contentEqualsIgnoreCase(mimeType)) {
            schema.setType("object");
            for (Signature.Parameter parameter : parameters) {
                String paramType = parameter.getContentType();
                Schema paramSchema = new Schema(parameter.getType(), parameter.getDescription());
                schema.addProperty(parameter.getName(), paramSchema, parameter.isRequired());
                if (paramType != null && !paramType.startsWith("text/")) {
                    paramSchema.setFormat("binary");
                    mediaType.addEncoding(parameter.getName(), new Encoding(paramType));
                }
            }
        } else {
            schema.setType("string");
            schema.setFormat("binary");
        }

        return mediaType;
    }
}
