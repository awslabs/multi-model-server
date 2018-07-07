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

import com.amazonaws.ml.mms.archive.Signature;
import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.wlm.Model;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpUtil;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public final class OpenApiUtils {

    private OpenApiUtils() {}

    public static String getMetadata(String host, String protocol, Map<String, Model> models) {
        Swagger swagger = new Swagger();
        Info info = new Info();
        info.setTitle("Model Serving APIs");
        info.setDescription(
                "Model Server is a flexible and easy to use tool for serving deep learning models");
        info.setVersion("1.0.0");
        swagger.setInfo(info);

        swagger.setHost(host);
        swagger.addScheme(protocol);

        swagger.addPath("/api-description", getApiDescriptionPath());
        swagger.addPath("/ping", getPingPath());
        swagger.addPath("/invocations", getInvocationsPath());
        swagger.addPath("/register", getRegisterModelPath());
        swagger.addPath("/unregister", getUnRegisterModelPath());
        swagger.addPath("/models", getListModelsPath());
        swagger.addPath("/model", getDescribeModelPath());
        swagger.addPath("/groups", getListGroupsPath());
        swagger.addPath("/scale", getScaleWorkerPath());

        List<String> modelNames = new ArrayList<>(models.keySet());
        Collections.sort(modelNames);

        for (String modelName : modelNames) {
            Signature signature = models.get(modelName).getModelArchive().getSignature();
            swagger.addPath(modelName + "/predict", getModelPath(modelName, signature));
        }

        return JsonUtils.GSON_PRETTY.toJson(swagger);
    }

    private static Path getApiDescriptionPath() {
        Response resp = new Response("200", "A swagger 2.0 json descriptor.");

        ObjectProperty schema = new ObjectProperty();
        resp.setSchema(schema);

        Operation operation = new Operation("openApiDescription");
        operation.addResponse(resp);
        operation.addProduce(HttpHeaderValues.APPLICATION_JSON.toString());

        Path path = new Path();
        path.setGet(operation);
        return path;
    }

    private static Path getPingPath() {
        Response resp = new Response("200", "Model server status.");

        ObjectProperty schema = new ObjectProperty();
        schema.addProperty(new StringProperty("status", "Overall status of the Model Server."));
        resp.setSchema(schema);

        Operation operation = new Operation("ping");
        operation.addResponse(resp);
        operation.addProduce(HttpHeaderValues.APPLICATION_JSON.toString());

        Path path = new Path();
        path.setGet(operation);
        return path;
    }

    private static Path getInvocationsPath() {
        Operation operation =
                new Operation("invocations", "A generic invocation entry point for all models.");

        FormParameter param = new FormParameter();
        param.setName("modelName");
        param.setRequired(true);
        param.setDescription("Name of model");
        param.setType("string");
        operation.addParameter(param);

        param = new FormParameter();
        param.setName("data");
        param.setRequired(true);
        param.setDescription("Inference input data");
        param.setType("file");
        operation.addParameter(param);

        Response resp = new Response("200", "OK");
        resp.setSchema(new ObjectProperty());

        operation.addResponse(resp);
        operation.addConsume(HttpHeaderValues.MULTIPART_FORM_DATA.toString());

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getRegisterModelPath() {
        Operation operation =
                new Operation("registerModel", "Register a new model in Model Server.");
        operation.addConsume(HttpHeaderValues.APPLICATION_JSON.toString());

        Schema schema = new Schema("object");
        schema.addProperty(new StringProperty("modelName", true, "Name of model to register."));
        schema.addProperty(
                new StringProperty(
                        "modelUrl",
                        true,
                        "Model archive download url, support local file or HTTP(s) protocol. For S3, consider use pre-signed url."));

        BodyParameter param = new BodyParameter("Model", schema);
        operation.addParameter(param);

        operation.addResponse(new Response("200", "Register success"));
        operation.addResponse(new Response("400", "Invalid model URL."));
        operation.addResponse(new Response("404", "Unable to download model archive"));
        operation.addResponse(
                new Response("400", "Invalid model archive file format, expecting a zip file."));
        operation.addResponse(new Response("400", "Unable to parse model archive manifest file."));
        operation.addResponse(
                new Response("400", "Unable to open dependent files specified in manifest file."));

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getUnRegisterModelPath() {
        Operation operation =
                new Operation("unregisterModel", "Unregister a model from Model Server.");
        operation.addConsume(HttpHeaderValues.APPLICATION_JSON.toString());

        Schema schema = new Schema("object");
        schema.addProperty(new StringProperty("modelName", true, "Name of model to register."));
        schema.addProperty(
                new Property("boolean", "forced", "Force terminate backend worker process."));

        BodyParameter param = new BodyParameter("model", schema);
        operation.addParameter(param);

        operation.addResponse(new Response("200", "Unregister success"));
        operation.addResponse(new Response("404", "Model not found."));

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getListModelsPath() {
        Operation operation =
                new Operation("listModels", "List registered models in Model Server.");

        Schema requestSchema = new Schema("object");
        Property property =
                new IntegerProperty(
                        "Limit",
                        "Use this parameter to specify the maximum number of items to return. When this value is present, Model Server does not return more than the specified number of items, but it might return fewer. This value is optional. If you include a value, it must be between 1 and 1000, inclusive. If you do not include a value, it defaults to 100.");
        property.setDefaultValue("100");
        requestSchema.addProperty(property);

        property =
                new StringProperty(
                        "NextPageToken",
                        "The token to retrieve the next set of results. Model Server provides the token when the response from a previous call has more results than the maximum page size.");
        requestSchema.addProperty(property);

        BodyParameter param = new BodyParameter("pagination", requestSchema);
        operation.addParameter(param);

        Response okResp = new Response("200", "OK");

        ObjectProperty responseSchema = new ObjectProperty();
        responseSchema.addProperty(
                new StringProperty(
                        "NextPageToken",
                        "Use this parameter in a subsequent request after you receive a response with truncated results. Set it to the value of NextMarker from the truncated response you just received."));
        ArrayProperty modelsProp = new ArrayProperty("models", "A list of registered models.");
        ObjectProperty modelProp = new ObjectProperty();
        modelProp.addProperty(new StringProperty("modelName", "Name of the model."));
        modelProp.addProperty(
                new IntegerProperty("modelGroupId", "Model group that the model belongs to."));
        modelProp.addProperty(new StringProperty("modelHash", "SHA1 hash of the model."));
        modelProp.addProperty(new StringProperty("modelUrl", "URL of the model."));
        modelsProp.setItems(modelProp);

        responseSchema.addProperty(modelsProp);
        okResp.setSchema(responseSchema);

        operation.addResponse(okResp);
        operation.addConsume(HttpHeaderValues.APPLICATION_JSON.toString());
        operation.addProduce(HttpHeaderValues.APPLICATION_JSON.toString());

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getDescribeModelPath() {
        Operation operation =
                new Operation(
                        "describeModel",
                        "Provides detailed information about the specified model.");

        Schema requestSchema = new Schema("object");
        requestSchema.addProperty(
                new StringProperty("modelName", true, "Name of model to describe."));

        BodyParameter param = new BodyParameter("model", requestSchema);
        operation.addParameter(param);

        Response okResp = new Response("200", "OK");

        ObjectProperty responseSchema = new ObjectProperty();
        responseSchema.addProperty(new StringProperty("modelName", "The name of the model."));
        responseSchema.addProperty(
                new IntegerProperty("modelGroupId", "Model group id that the model belongs to."));
        responseSchema.addProperty(
                new IntegerProperty(
                        "rejectedRequests",
                        "Number requests has been rejected in last 10 minutes."));
        responseSchema.addProperty(
                new IntegerProperty("waitingQueueSize", "Number requests waiting in the queue."));
        responseSchema.addProperty(new IntegerProperty("batchSize", "Configured batch size."));
        responseSchema.addProperty(
                new IntegerProperty("batchDelay", "Configured batch delay in ms."));
        responseSchema.addProperty(new IntegerProperty("requestPerSecond", "Request per second."));

        ArrayProperty modelsProp =
                new ArrayProperty("workers", "A list of workers that serving the models.");
        ObjectProperty modelProp = new ObjectProperty();
        modelProp.addProperty(new StringProperty("type", "GPU or CPU"));
        modelProp.addProperty(new StringProperty("status", "Health status of the worker process"));
        modelsProp.setItems(modelProp);

        responseSchema.addProperty(modelsProp);
        okResp.setSchema(responseSchema);

        operation.addResponse(okResp);
        operation.addConsume(HttpHeaderValues.APPLICATION_JSON.toString());
        operation.addProduce(HttpHeaderValues.APPLICATION_JSON.toString());

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getListGroupsPath() {
        Operation operation = new Operation("listGroups", "List registered model groups.");

        Response okResp = new Response("200", "OK");

        ObjectProperty responseSchema = new ObjectProperty();
        ArrayProperty modelsProp =
                new ArrayProperty("groups", "A list of registered model groups.");
        ObjectProperty modelProp = new ObjectProperty();
        modelProp.addProperty(new IntegerProperty("modelGroupId", "Model group ID."));
        modelProp.addProperty(
                new StringProperty("minWorker", "Configured minimum number of worker processes."));
        modelProp.addProperty(
                new StringProperty("maxWorker", "Configured maximum number of worker processes."));
        modelProp.addProperty(
                new StringProperty(
                        "numberGpu", "Configured number of GPU worker processes to create."));
        modelProp.addProperty(
                new StringProperty("activeWorkers", "Active running worker processes."));
        modelProp.addProperty(new StringProperty("status", "Status of the model group."));
        modelsProp.setItems(modelProp);

        responseSchema.addProperty(modelsProp);
        okResp.setSchema(responseSchema);

        operation.addResponse(okResp);
        operation.addProduce(HttpHeaderValues.APPLICATION_JSON.toString());

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getScaleWorkerPath() {
        Operation operation =
                new Operation(
                        "setAutoScale",
                        "Configure number of workers for a model group, This is a asynchronized call.");
        operation.addConsume(HttpHeaderValues.APPLICATION_JSON.toString());

        Schema schema = new Schema("object");
        schema.addProperty(
                new IntegerProperty("modelGroupId", true, "ID of model group to scale."));
        schema.addProperty(
                new IntegerProperty("minWorker", true, "Minimum number of worker processes."));
        schema.addProperty(
                new IntegerProperty("maxWorker", true, "Maximum number of worker processes."));
        schema.addProperty(
                new IntegerProperty(
                        "numberGpu", true, "Number of GPU worker processes to create."));
        schema.addProperty(
                new IntegerProperty(
                        "timeout",
                        "Waiting up to the specified wait time if necessary for a worker to complete all pending requests. Use 0 to terminate backend worker process immediately. Use -1 for wait infinitely."));

        BodyParameter param = new BodyParameter("scaleParam", schema);
        operation.addParameter(param);

        operation.addResponse(new Response("200", "Scale configuration set."));
        operation.addResponse(new Response("404", "Group not found."));

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getModelPath(String modelName, Signature signature) {
        Operation operation =
                new Operation(
                        modelName + "_predict",
                        "A predict entry point for model: " + modelName + '.');
        Signature.Request req = signature.getRequest();
        String contentType = req.getContentType();
        if (contentType != null) {
            operation.addConsume(req.getContentType());

            CharSequence mimeType = HttpUtil.getMimeType(contentType);
            List<Signature.Shape> inputs = req.getInputs();
            if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(mimeType)) {
                Schema schema = new Schema("object");
                for (Signature.Shape shape : inputs) {
                    schema.addProperty(
                            new StringProperty(
                                    shape.getName(), shape.isRequired(), shape.getDescription()));
                }
                BodyParameter param = new BodyParameter("body", schema);
                operation.addParameter(param);
            } else if (HttpHeaderValues.MULTIPART_FORM_DATA.contentEqualsIgnoreCase(mimeType)) {
                for (Signature.Shape shape : inputs) {
                    FormParameter param = new FormParameter();
                    param.setName(shape.getName());
                    param.setRequired(shape.isRequired());
                    param.setDescription(shape.getDescription());
                    param.setType(shape.getType());
                    operation.addParameter(param);
                }
            }
        }

        Signature.Response resp = signature.getResponse();
        contentType = resp.getContentType();
        if (contentType != null) {
            operation.addProduce(contentType);
        }

        Response swaggerResponse = new Response("200", "OK");

        operation.addResponse(swaggerResponse);

        Path path = new Path();
        path.setPost(operation);
        return path;
    }
}
