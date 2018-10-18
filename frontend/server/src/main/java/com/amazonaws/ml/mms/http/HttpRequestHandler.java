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
package com.amazonaws.ml.mms.http;

import com.amazonaws.ml.mms.archive.InvalidModelException;
import com.amazonaws.ml.mms.archive.Manifest;
import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.common.ErrorCodes;
import com.amazonaws.ml.mms.openapi.OpenApiUtils;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.messages.InputParameter;
import com.amazonaws.ml.mms.util.messages.RequestInput;
import com.amazonaws.ml.mms.util.messages.WorkerCommands;
import com.amazonaws.ml.mms.wlm.Job;
import com.amazonaws.ml.mms.wlm.Model;
import com.amazonaws.ml.mms.wlm.ModelManager;
import com.amazonaws.ml.mms.wlm.WorkerInitializationException;
import com.amazonaws.ml.mms.wlm.WorkerThread;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests.
 *
 * <p>This class
 */
public class HttpRequestHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    private static final Logger logger = LoggerFactory.getLogger(HttpRequestHandler.class);

    /** Creates a new {@code HttpRequestHandler} instance. */
    public HttpRequestHandler() {}

    /** {@inheritDoc} */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest req) {
        NettyUtils.requestReceived(ctx.channel(), req);
        if (!req.decoderResult().isSuccess()) {
            NettyUtils.sendError(
                    ctx, HttpResponseStatus.BAD_REQUEST, ErrorCodes.MESSAGE_DECODE_FAILURE);
            return;
        }

        try {
            handleRequest(ctx, req);
        } catch (IllegalArgumentException e) {
            logger.debug("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, e.getMessage());
        } catch (Throwable t) {
            logger.error("", t);
            NettyUtils.sendError(
                    ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, ErrorCodes.UNKNOWN_ERROR);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        logger.error("", cause);
        ctx.close();
    }

    private void handleRequest(ChannelHandlerContext ctx, FullHttpRequest req) {
        QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
        String path = decoder.path();
        if ("/".equals(path)) {
            handleListModels(ctx, req);
            return;
        }

        String[] segments = decoder.path().split("/");
        switch (segments[1]) {
            case "ping":
                handlePing(ctx);
                break;
            case "api-description":
                NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listApis());
                break;
            case "invocations":
                handleInvocations(ctx, req, decoder);
                break;
            case "predictions":
                handlePredictions(ctx, req, segments);
                break;
            case "models":
                handleModelsApi(ctx, req, segments, decoder);
                break;
            default:
                NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, ErrorCodes.INVALID_URI);
                break;
        }
    }

    private void handleListModels(ChannelHandlerContext ctx, FullHttpRequest req) {
        if (HttpMethod.OPTIONS.equals(req.method())) {
            NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listApis());
            return;
        }
        NettyUtils.sendError(
                ctx, HttpResponseStatus.NOT_FOUND, ErrorCodes.LIST_MODELS_INVALID_REQUEST_HEADER);
    }

    private void handlePing(ChannelHandlerContext ctx) {
        ModelManager.getInstance().workerStatus(ctx);
    }

    private void handleModelsApi(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            String[] segments,
            QueryStringDecoder decoder) {
        HttpMethod method = req.method();
        if (segments.length < 3) {
            if (HttpMethod.GET.equals(method)) {
                handleListModels(ctx, decoder);
                return;
            } else if (HttpMethod.POST.equals(method)) {
                handleRegisterModel(ctx, decoder);
                return;
            }
            NettyUtils.sendError(
                    ctx,
                    HttpResponseStatus.BAD_REQUEST,
                    ErrorCodes.MODELS_API_INVALID_MODELS_REQUEST);
        }

        if (HttpMethod.GET.equals(method)) {
            handleDescribeModel(ctx, segments[2]);
        } else if (HttpMethod.PUT.equals(method)) {
            handleScaleModel(ctx, decoder, segments[2]);
        } else if (HttpMethod.DELETE.equals(method)) {
            handleUnregisterModel(ctx, segments[2]);
        } else {
            NettyUtils.sendError(
                    ctx,
                    HttpResponseStatus.BAD_REQUEST,
                    ErrorCodes.MODELS_API_INVALID_MODELS_REQUEST);
        }
    }

    private void handlePredictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments) {
        if (segments.length < 3) {
            NettyUtils.sendError(
                    ctx,
                    HttpResponseStatus.BAD_REQUEST,
                    ErrorCodes.PREDICTIONS_API_INVALID_REQUEST);
            return;
        }
        String modelName = segments[2];

        ModelManager modelManager = ModelManager.getInstance();
        Model model = modelManager.getModels().get(modelName);
        if (model == null) {
            NettyUtils.sendError(
                    ctx,
                    HttpResponseStatus.NOT_FOUND,
                    ErrorCodes.PREDICTIONS_API_MODEL_NOT_REGISTERED);
            return;
        }

        if (HttpMethod.OPTIONS.equals(req.method())) {
            String resp = OpenApiUtils.getModelApi(model);
            NettyUtils.sendJsonResponse(ctx, resp);
            return;
        }

        RequestInput input;
        try {
            input = parseRequest(ctx, req);
        } catch (IllegalArgumentException e) {
            NettyUtils.sendError(
                    ctx,
                    HttpResponseStatus.BAD_REQUEST,
                    ErrorCodes.PREDICTIONS_API_INVALID_PARAMETERS);
            return;
        }

        Job job = new Job(ctx, modelName, WorkerCommands.PREDICT, input);
        HttpResponseStatus status = ModelManager.getInstance().addJob(job);
        if (status != HttpResponseStatus.OK) {
            String code;
            if (status == HttpResponseStatus.NOT_FOUND) {
                code = ErrorCodes.PREDICTIONS_API_MODEL_NOT_REGISTERED;
            } else {
                code = ErrorCodes.PREDICTIONS_API_MODEL_NOT_SCALED;
            }
            NettyUtils.sendError(ctx, status, code);
        }
    }

    private void handleInvocations(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder) {
        String modelName = NettyUtils.getParameter(decoder, "model_name", null);

        HttpMethod method = req.method();
        if (!HttpMethod.POST.equals(method)) {
            NettyUtils.sendError(
                    ctx,
                    HttpResponseStatus.BAD_REQUEST,
                    ErrorCodes.PREDICTIONS_API_INVALID_REQUEST);
            return;
        }

        RequestInput input;
        try {
            input = parseRequest(ctx, req);
            if (modelName == null) {
                modelName = input.getStringParameter("model_name");
            }
        } catch (IllegalArgumentException e) {
            NettyUtils.sendError(
                    ctx,
                    HttpResponseStatus.BAD_REQUEST,
                    ErrorCodes.PREDICTIONS_API_INVALID_PARAMETERS);
            return;
        }

        Job job = new Job(ctx, modelName, WorkerCommands.PREDICT, input);
        HttpResponseStatus status = ModelManager.getInstance().addJob(job);
        if (status != HttpResponseStatus.OK) {
            String code;
            if (status == HttpResponseStatus.NOT_FOUND) {
                code = ErrorCodes.PREDICTIONS_API_MODEL_NOT_REGISTERED;
            } else {
                code = ErrorCodes.PREDICTIONS_API_MODEL_NOT_SCALED;
            }
            NettyUtils.sendError(ctx, status, code);
        }
    }

    private void handleListModels(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
        int pageToken = NettyUtils.getIntParameter(decoder, "nextPageToken", 0);
        if (limit > 100 || limit < 0) {
            limit = 100;
        }
        if (pageToken < 0) {
            pageToken = 0;
        }

        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Model> models = modelManager.getModels();

        List<String> keys = new ArrayList<>(models.keySet());
        Collections.sort(keys);
        ListModelsResponse list = new ListModelsResponse();

        int last = pageToken + limit;
        if (last > keys.size()) {
            last = keys.size();
        } else {
            list.setNextPageToken(String.valueOf(last));
        }

        for (int i = pageToken; i < last; ++i) {
            String modelName = keys.get(i);
            Model model = models.get(modelName);
            list.addModel(modelName, model.getModelUrl());
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleRegisterModel(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        String modelUrl = NettyUtils.getParameter(decoder, "url", null);
        if (modelUrl == null) {
            NettyUtils.sendError(
                    ctx, HttpResponseStatus.BAD_REQUEST, ErrorCodes.MODELS_POST_INVALID_REQUEST);
            return;
        }

        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        String runtime = NettyUtils.getParameter(decoder, "runtime", null);
        String handler = NettyUtils.getParameter(decoder, "handler", null);
        int batchSize = NettyUtils.getIntParameter(decoder, "batch_size", 1);
        int maxBatchDelay = NettyUtils.getIntParameter(decoder, "max_batch_delay", 100);
        int initialWorkers = NettyUtils.getIntParameter(decoder, "initial_workers", 0);
        boolean synchronous =
                Boolean.parseBoolean(NettyUtils.getParameter(decoder, "synchronous", null));

        Manifest.RuntimeType runtimeType = null;
        if (runtime != null) {
            try {
                runtimeType = Manifest.RuntimeType.fromValue(runtime);
            } catch (IllegalArgumentException e) {
                String msg = e.getMessage();
                NettyUtils.sendError(
                        ctx,
                        HttpResponseStatus.BAD_REQUEST,
                        ErrorCodes.MODELS_POST_MODEL_MANIFEST_RUNTIME_INVALID
                                + " Invalid model runtime given. "
                                + msg);
                return;
            }
        }

        ModelManager modelManager = ModelManager.getInstance();
        final ModelArchive archive;
        try {
            archive =
                    modelManager.registerModel(
                            modelUrl, modelName, runtimeType, handler, batchSize, maxBatchDelay);
        } catch (InvalidModelException e) {
            logger.warn("Failed to load model", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, e.getErrorCode());
            return;
        }

        modelName = archive.getModelName();

        final String msg = "Model \"" + modelName + "\" registered";
        if (initialWorkers <= 0) {
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
            return;
        }

        updateModelWorkers(
                ctx,
                modelName,
                initialWorkers,
                initialWorkers,
                synchronous,
                f -> {
                    try {
                        modelManager.unregisterModel(archive.getModelName());
                    } catch (WorkerInitializationException ignore) {
                        // ignore
                    }
                    archive.clean();
                    return null;
                });
    }

    private void handleUnregisterModel(ChannelHandlerContext ctx, String modelName) {
        ModelManager modelManager = ModelManager.getInstance();
        try {
            if (!modelManager.unregisterModel(modelName)) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, "Model not found");
            }
        } catch (WorkerInitializationException e) {
            logger.warn("Failed to unregister model.", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, e.getErrorCode());
            return;
        }
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName) {
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", 1);
        boolean synchronous =
                Boolean.parseBoolean(NettyUtils.getParameter(decoder, "synchronous", null));

        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.getModels().containsKey(modelName)) {
            NettyUtils.sendError(
                    ctx, HttpResponseStatus.NOT_FOUND, ErrorCodes.MODELS_API_MODEL_NOT_FOUND);
            return;
        }
        updateModelWorkers(ctx, modelName, minWorkers, maxWorkers, synchronous, null);
    }

    private void handleDescribeModel(ChannelHandlerContext ctx, String modelName) {
        ModelManager modelManager = ModelManager.getInstance();
        Model model = modelManager.getModels().get(modelName);
        if (model == null) {
            NettyUtils.sendError(
                    ctx, HttpResponseStatus.NOT_FOUND, ErrorCodes.MODELS_API_MODEL_NOT_FOUND);
            return;
        }

        DescribeModelResponse resp = new DescribeModelResponse();
        resp.setModelName(modelName);
        resp.setModelUrl(model.getModelUrl());
        resp.setBatchSize(model.getBatchSize());
        resp.setMaxBatchDelay(model.getMaxBatchDelay());
        resp.setMaxWorkers(model.getMaxWorkers());
        resp.setMinWorkers(model.getMinWorkers());
        Manifest manifest = model.getModelArchive().getManifest();
        Manifest.Engine engine = manifest.getEngine();
        if (engine != null) {
            resp.setEngine(engine.getEngineName().getValue());
        }
        resp.setModelVersion(manifest.getModel().getModelVersion());
        resp.setRuntime(manifest.getRuntime().getValue());

        List<WorkerThread> workers = modelManager.getWorkers(modelName);
        for (WorkerThread worker : workers) {
            String workerId = worker.getName();
            long startTime = worker.getStartTime();
            boolean isRunning = worker.isRunning();
            int gpuId = worker.getGpuId();
            long memory = worker.getMemory();
            resp.addWorker(workerId, startTime, isRunning, gpuId, memory);
        }

        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private static RequestInput parseRequest(ChannelHandlerContext ctx, FullHttpRequest req) {
        String requestId = NettyUtils.getRequestId(ctx.channel());
        RequestInput inputData = new RequestInput(requestId);
        CharSequence contentType = HttpUtil.getMimeType(req);
        if (HttpPostRequestDecoder.isMultipart(req)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)) {
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(req);
            try {
                while (form.hasNext()) {
                    inputData.addParameter(NettyUtils.getFormData(form.next()));
                }
            } catch (HttpPostRequestDecoder.EndOfDataDecoderException ignore) {
                logger.debug("End of multipart items.");
            } finally {
                form.cleanFiles();
            }
        } else {
            byte[] content = NettyUtils.getBytes(req.content());
            inputData.addParameter(new InputParameter("body", content, contentType));
        }
        return inputData;
    }

    private void updateModelWorkers(
            final ChannelHandlerContext ctx,
            String modelName,
            int minWorkers,
            int maxWorkers,
            boolean synchronous,
            final Function<Void, Void> onError) {
        ModelManager modelManager = ModelManager.getInstance();
        try {
            CompletableFuture<Boolean> future =
                    modelManager.updateModel(modelName, minWorkers, maxWorkers);
            if (!synchronous) {
                NettyUtils.sendJsonResponse(
                        ctx, new StatusResponse("Worker updated"), HttpResponseStatus.ACCEPTED);
                return;
            }
            future.thenApply(
                            v -> {
                                if (!v) {
                                    if (onError != null) {
                                        onError.apply(null);
                                    }
                                    NettyUtils.sendError(
                                            ctx,
                                            HttpResponseStatus.BAD_REQUEST,
                                            ErrorCodes.MODELS_API_MODEL_NOT_FOUND);
                                } else {
                                    NettyUtils.sendJsonResponse(
                                            ctx,
                                            new StatusResponse("Worker scaled"),
                                            HttpResponseStatus.OK);
                                }
                                return v;
                            })
                    .exceptionally(
                            (e) -> {
                                if (onError != null) {
                                    onError.apply(null);
                                }
                                NettyUtils.sendError(
                                        ctx,
                                        HttpResponseStatus.INTERNAL_SERVER_ERROR,
                                        e.getMessage());
                                return null;
                            });
        } catch (WorkerInitializationException e) {
            logger.error("Failed update model workers.", e);
            if (onError != null) {
                onError.apply(null);
            }
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, e.getMessage());
        }
    }
}
