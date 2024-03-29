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

import com.amazonaws.ml.mms.archive.Manifest;
import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.archive.ModelException;
import com.amazonaws.ml.mms.archive.ModelNotFoundException;
import com.amazonaws.ml.mms.http.messages.RegisterModelRequest;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.wlm.Model;
import com.amazonaws.ml.mms.wlm.ModelManager;
import com.amazonaws.ml.mms.wlm.WorkerThread;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class ManagementRequestHandler extends HttpRequestHandlerChain {

    /** Creates a new {@code ManagementRequestHandler} instance. */
    public ManagementRequestHandler(Map<String, ModelServerEndpoint> ep) {
        endpointMap = ep;
    }

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        if (isManagementReq(segments)) {
            if (endpointMap.getOrDefault(segments[1], null) != null) {
                handleCustomEndpoint(ctx, req, segments, decoder);
            } else {
                if (!"models".equals(segments[1])) {
                    throw new ResourceNotFoundException();
                }

                HttpMethod method = req.method();
                if (segments.length < 3) {
                    if (HttpMethod.GET.equals(method)) {
                        handleListModels(ctx, decoder);
                        return;
                    } else if (HttpMethod.POST.equals(method)) {
                        handleRegisterModel(ctx, decoder, req);
                        return;
                    }
                    throw new MethodNotAllowedException();
                }

                if (HttpMethod.GET.equals(method)) {
                    handleDescribeModel(ctx, segments[2]);
                } else if (HttpMethod.PUT.equals(method)) {
                    handleScaleModel(ctx, decoder, segments[2]);
                } else if (HttpMethod.DELETE.equals(method)) {
                    handleUnregisterModel(ctx, segments[2]);
                } else {
                    throw new MethodNotAllowedException();
                }
            }
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private boolean isManagementReq(String[] segments) {
        return segments.length == 0
                || ((segments.length == 2 || segments.length == 3) && segments[1].equals("models"))
                || endpointMap.containsKey(segments[1]);
    }

    private void handleListModels(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
        int pageToken = NettyUtils.getIntParameter(decoder, "next_page_token", 0);
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

    private void handleDescribeModel(ChannelHandlerContext ctx, String modelName)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        Model model = modelManager.getModels().get(modelName);
        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }

        DescribeModelResponse resp = new DescribeModelResponse();
        resp.setModelName(modelName);
        resp.setModelUrl(model.getModelUrl());
        resp.setBatchSize(model.getBatchSize());
        resp.setMaxBatchDelay(model.getMaxBatchDelay());
        resp.setMaxWorkers(model.getMaxWorkers());
        resp.setMinWorkers(model.getMinWorkers());
        resp.setLoadedAtStartup(modelManager.getStartupModels().contains(modelName));
        Manifest manifest = model.getModelArchive().getManifest();
        Manifest.Engine engine = manifest.getEngine();
        if (engine != null) {
            resp.setEngine(engine.getEngineName());
        }
        resp.setModelVersion(manifest.getModel().getModelVersion());
        resp.setRuntime(manifest.getRuntime().getValue());

        List<WorkerThread> workers = modelManager.getWorkers(modelName);
        for (WorkerThread worker : workers) {
            String workerId = worker.getWorkerId();
            long startTime = worker.getStartTime();
            boolean isRunning = worker.isRunning();
            int gpuId = worker.getGpuId();
            long memory = worker.getMemory();
            resp.addWorker(workerId, startTime, isRunning, gpuId, memory);
        }

        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void handleRegisterModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req)
            throws ModelException {
        RegisterModelRequest registerModelRequest = parseRequest(req, decoder);
        String modelUrl = registerModelRequest.getModelUrl();
        if (modelUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }

        String modelName = registerModelRequest.getModelName();
        String runtime = registerModelRequest.getRuntime();
        String handler = registerModelRequest.getHandler();
        int batchSize = registerModelRequest.getBatchSize();
        int maxBatchDelay = registerModelRequest.getMaxBatchDelay();
        int initialWorkers = registerModelRequest.getInitialWorkers();
        boolean synchronous = registerModelRequest.isSynchronous();
        int responseTimeoutSeconds = registerModelRequest.getResponseTimeoutSeconds();
        String preloadModel = registerModelRequest.getPreloadModel();
        if (preloadModel == null) {
            preloadModel = ConfigManager.getInstance().getPreloadModel();
        }
        if (responseTimeoutSeconds == -1) {
            responseTimeoutSeconds = ConfigManager.getInstance().getDefaultResponseTimeoutSeconds();
        }
        Manifest.RuntimeType runtimeType = null;
        if (runtime != null) {
            try {
                runtimeType = Manifest.RuntimeType.fromValue(runtime);
            } catch (IllegalArgumentException e) {
                throw new BadRequestException(e);
            }
        }

        ModelManager modelManager = ModelManager.getInstance();
        final ModelArchive archive;
        try {

            archive =
                    modelManager.registerModel(
                            modelUrl,
                            modelName,
                            runtimeType,
                            handler,
                            batchSize,
                            maxBatchDelay,
                            responseTimeoutSeconds,
                            null,
                            preloadModel);
        } catch (IOException | InterruptedException | ExecutionException | TimeoutException e) {
            throw new InternalServerException("Failed to save model: " + modelUrl, e);
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
                    modelManager.unregisterModel(archive.getModelName());
                    return null;
                });
    }

    private void handleUnregisterModel(ChannelHandlerContext ctx, String modelName)
            throws ModelNotFoundException, InternalServerException, RequestTimeoutException {
        ModelManager modelManager = ModelManager.getInstance();
        HttpResponseStatus httpResponseStatus = modelManager.unregisterModel(modelName);
        if (httpResponseStatus == HttpResponseStatus.NOT_FOUND) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR) {
            throw new InternalServerException("Interrupted while cleaning resources: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.REQUEST_TIMEOUT) {
            throw new RequestTimeoutException("Timed out while cleaning resources: " + modelName);
        }
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName)
            throws ModelNotFoundException {
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", minWorkers);
        if (maxWorkers < minWorkers) {
            throw new BadRequestException("max_worker cannot be less than min_worker.");
        }
        boolean synchronous =
                Boolean.parseBoolean(NettyUtils.getParameter(decoder, "synchronous", null));

        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.getModels().containsKey(modelName)) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        updateModelWorkers(ctx, modelName, minWorkers, maxWorkers, synchronous, null);
    }

    private void updateModelWorkers(
            final ChannelHandlerContext ctx,
            final String modelName,
            int minWorkers,
            int maxWorkers,
            boolean synchronous,
            final Function<Void, Void> onError) {

        ModelManager modelManager = ModelManager.getInstance();
        CompletableFuture<HttpResponseStatus> future =
                modelManager.updateModel(modelName, minWorkers, maxWorkers);
        if (!synchronous) {
            NettyUtils.sendJsonResponse(
                    ctx,
                    new StatusResponse("Processing worker updates..."),
                    HttpResponseStatus.ACCEPTED);
            return;
        }
        future.thenApply(
                        v -> {
                            boolean status = modelManager.scaleRequestStatus(modelName);
                            if (HttpResponseStatus.OK.equals(v)) {
                                if (status) {
                                    NettyUtils.sendJsonResponse(
                                            ctx, new StatusResponse("Workers scaled"), v);
                                } else {
                                    NettyUtils.sendJsonResponse(
                                            ctx,
                                            new StatusResponse("Workers scaling in progress..."),
                                            new HttpResponseStatus(210, "Partial Success"));
                                }
                            } else {
                                NettyUtils.sendError(
                                        ctx,
                                        v,
                                        new InternalServerException("Failed to start workers"));
                                if (onError != null) {
                                    onError.apply(null);
                                }
                            }
                            return v;
                        })
                .exceptionally(
                        (e) -> {
                            if (onError != null) {
                                onError.apply(null);
                            }
                            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, e);
                            return null;
                        });
    }

    private RegisterModelRequest parseRequest(FullHttpRequest req, QueryStringDecoder decoder) {
        RegisterModelRequest in;
        CharSequence mime = HttpUtil.getMimeType(req);
        if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(mime)) {
            in =
                    JsonUtils.GSON.fromJson(
                            req.content().toString(CharsetUtil.UTF_8), RegisterModelRequest.class);
        } else {
            in = new RegisterModelRequest(decoder);
        }
        return in;
    }
}
