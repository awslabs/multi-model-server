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
import com.amazonaws.ml.mms.openapi.OpenApiUtils;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.messages.ModelInputs;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import com.amazonaws.ml.mms.wlm.Job;
import com.amazonaws.ml.mms.wlm.Model;
import com.amazonaws.ml.mms.wlm.ModelManager;
import com.amazonaws.ml.mms.wlm.WorkerInitializationException;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
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
        if (!req.decoderResult().isSuccess()) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        try {
            handleRequest(ctx, req);
        } catch (Throwable t) {
            logger.error("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR);
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
                NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND);
                break;
        }
    }

    private void handleListModels(ChannelHandlerContext ctx, FullHttpRequest req) {
        if (HttpMethod.OPTIONS.equals(req.method())) {
            NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listApis());
            return;
        }
        NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND);
    }

    private void handlePing(ChannelHandlerContext ctx) {
        NettyUtils.sendJsonResponse(ctx, new StatusResponse("healthy"));
    }

    private void handleModelsApi(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            String[] segments,
            QueryStringDecoder decoder) {
        HttpMethod method = req.method();
        if (segments.length < 3) {
            if (HttpMethod.GET.equals(method)) {
                handleListModels(ctx);
                return;
            } else if (HttpMethod.POST.equals(method)) {
                handleRegisterModel(ctx, decoder);
                return;
            }
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
        }

        if (HttpMethod.GET.equals(method)) {
            handleDescribeModel(ctx);
        } else if (HttpMethod.PUT.equals(method)) {
            handleScaleModel(ctx, decoder, segments[2]);
        } else if (HttpMethod.DELETE.equals(method)) {
            handleUnregisterModel(ctx, segments[2]);
        } else {
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND);
        }
    }

    private void handlePredictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments) {
        if (segments.length < 3) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }
        String modelName = segments[2];

        ModelManager modelManager = ModelManager.getInstance();
        Model model = modelManager.getModels().get(modelName);
        if (model == null) {
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND);
            return;
        }

        if (HttpMethod.OPTIONS.equals(req.method())) {
            String resp = OpenApiUtils.getModelApi(model);
            NettyUtils.sendJsonResponse(ctx, resp);
            return;
        }

        RequestBatch input;
        try {
            input = parseRequest(req);
        } catch (IllegalArgumentException e) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        Job job = new Job(ctx, modelName, "predict", input);
        HttpResponseStatus status = ModelManager.getInstance().addJob(job);
        if (status != HttpResponseStatus.OK) {
            NettyUtils.sendError(ctx, status);
        }
    }

    private void handleInvocations(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder) {
        String modelName = NettyUtils.getParameter(decoder, "model_name", null);

        HttpMethod method = req.method();
        if (!HttpMethod.POST.equals(method)) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        RequestBatch input;
        try {
            input = parseRequest(req);
            if (modelName == null) {
                modelName = input.getStringParameter("model_name");
            }
        } catch (IllegalArgumentException e) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        Job job = new Job(ctx, modelName, "predict", input);
        HttpResponseStatus status = ModelManager.getInstance().addJob(job);
        if (status != HttpResponseStatus.OK) {
            NettyUtils.sendError(ctx, status);
        }
    }

    private void handleListModels(ChannelHandlerContext ctx) {
        // TODO:
        NettyUtils.sendJsonResponse(ctx, new StatusResponse("Coming soon"));
    }

    private void handleRegisterModel(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        String modelUrl = NettyUtils.getParameter(decoder, "url", null);
        if (modelUrl == null) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        String runtime = NettyUtils.getParameter(decoder, "runtime", null);
        String handler = NettyUtils.getParameter(decoder, "handler", null);
        int batchSize = NettyUtils.getIntParameter(decoder, "batch_size", 1);
        int maxBatchDelay = NettyUtils.getIntParameter(decoder, "max_batch_delay", 100);

        Manifest.RuntimeType runtimeType = null;
        if (runtime != null) {
            try {
                runtimeType = Manifest.RuntimeType.fromValue(runtime);
            } catch (IllegalArgumentException e) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, e.getMessage());
                return;
            }
        }

        ModelManager modelManager = ModelManager.getInstance();
        try {
            modelManager.registerModel(
                    modelUrl, modelName, runtimeType, handler, batchSize, maxBatchDelay);
        } catch (InvalidModelException e) {
            logger.warn("Failed to load model", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        NettyUtils.sendJsonResponse(ctx, new StatusResponse("Model registered"));
    }

    private void handleUnregisterModel(ChannelHandlerContext ctx, String modelName) {
        ModelManager modelManager = ModelManager.getInstance();
        try {
            if (!modelManager.unregisterModel(modelName)) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            }
        } catch (WorkerInitializationException e) {
            logger.warn("Failed to load model", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        NettyUtils.sendJsonResponse(ctx, new StatusResponse("Model unregistered"));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName) {
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", 1);

        ModelManager modelManager = ModelManager.getInstance();
        try {
            if (!modelManager.updateModel(modelName, minWorkers, maxWorkers)) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
                return;
            }
        } catch (WorkerInitializationException e) {
            logger.error("Failed update model.", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR);
            return;
        }

        NettyUtils.sendJsonResponse(
                ctx, new StatusResponse("Worker updated"), HttpResponseStatus.ACCEPTED);
    }

    private void handleDescribeModel(ChannelHandlerContext ctx) {
        // TODO:
        NettyUtils.sendJsonResponse(ctx, new StatusResponse("Coming soon"));
    }

    private static RequestBatch parseRequest(FullHttpRequest req) {
        RequestBatch inputData = new RequestBatch();
        CharSequence contentType = HttpUtil.getMimeType(req);
        inputData.setContentType(contentType.toString());
        if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(contentType)) {
            inputData.addModelInput(new ModelInputs("body", NettyUtils.getBytes(req.content())));
        } else if (HttpPostRequestDecoder.isMultipart(req)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)) {
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(req);
            try {
                while (form.hasNext()) {
                    inputData.addModelInput(NettyUtils.getFormData(form.next()));
                }
            } catch (HttpPostRequestDecoder.EndOfDataDecoderException ignore) {
                logger.debug("End of multipart items.");
            } finally {
                form.cleanFiles();
            }
        } else {
            inputData.addModelInput(new ModelInputs("body", NettyUtils.getBytes(req.content())));
        }
        return inputData;
    }
}
