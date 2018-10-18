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
import com.amazonaws.ml.mms.openapi.OpenApiUtils;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.wlm.Model;
import com.amazonaws.ml.mms.wlm.ModelManager;
import com.amazonaws.ml.mms.wlm.WorkerThread;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class ManagementRequestHandler extends HttpRequestHandler {

    /** Creates a new {@code ManagementRequestHandler} instance. */
    public ManagementRequestHandler() {}

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        if (!"models".equals(segments[1])) {
            throw new ResourceNotFoundException();
        }

        HttpMethod method = req.method();
        if (segments.length < 3) {
            if (HttpMethod.GET.equals(method)) {
                handleListModels(ctx, decoder);
                return;
            } else if (HttpMethod.POST.equals(method)) {
                handleRegisterModel(ctx, decoder);
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

    @Override
    protected void handleApiDescription(ChannelHandlerContext ctx) {
        NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listManagementApis());
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
        Manifest manifest = model.getModelArchive().getManifest();
        Manifest.Engine engine = manifest.getEngine();
        if (engine != null) {
            resp.setEngine(engine.getEngineName().getValue());
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

    private void handleRegisterModel(ChannelHandlerContext ctx, QueryStringDecoder decoder)
            throws ModelException {
        String modelUrl = NettyUtils.getParameter(decoder, "url", null);
        if (modelUrl == null) {
            throw new BadRequestException("Parameter url is required.");
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
                throw new BadRequestException(e);
            }
        }

        ModelManager modelManager = ModelManager.getInstance();
        final ModelArchive archive;
        try {
            archive =
                    modelManager.registerModel(
                            modelUrl, modelName, runtimeType, handler, batchSize, maxBatchDelay);
        } catch (IOException e) {
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
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.unregisterModel(modelName)) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName)
            throws ModelNotFoundException {
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", 1);
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
        CompletableFuture<Boolean> future =
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
                            String response = "Workers scaled";
                            HttpResponseStatus httpCode = HttpResponseStatus.OK;

                            if (!v) {
                                response = "Workers scaling in progress";
                                if (onError != null) {
                                    onError.apply(null);
                                    response = "Load failed... Deregistered model " + modelName;
                                }
                                NettyUtils.sendError(
                                        ctx,
                                        HttpResponseStatus.NOT_FOUND,
                                        new ModelNotFoundException(response));
                            } else {
                                if (!status) {
                                    response = "Workers scaling in progress..";
                                    httpCode = new HttpResponseStatus(210, "Partial Success");
                                }

                                NettyUtils.sendJsonResponse(
                                        ctx, new StatusResponse(response), httpCode);
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
}
