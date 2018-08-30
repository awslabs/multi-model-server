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

import com.amazonaws.ml.mms.common.ErrorCodes;
import com.amazonaws.ml.mms.openapi.OpenApiUtils;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.messages.InputParameter;
import com.amazonaws.ml.mms.util.messages.RequestInput;
import com.amazonaws.ml.mms.util.messages.WorkerCommands;
import com.amazonaws.ml.mms.wlm.Job;
import com.amazonaws.ml.mms.wlm.Model;
import com.amazonaws.ml.mms.wlm.ModelManager;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class InferenceRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);

    /** Creates a new {@code HttpRequestHandler} instance. */
    public InferenceRequestHandler() {}

    protected boolean handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        switch (segments[1]) {
            case "invocations":
                handleInvocations(ctx, req, decoder);
                return true;
            case "predictions":
                handlePredictions(ctx, req, segments);
                return true;
            default:
                return false;
        }
    }

    protected void handleApiDescription(ChannelHandlerContext ctx) {
        NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listInferenceApis());
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

    private static RequestInput parseRequest(ChannelHandlerContext ctx, FullHttpRequest req) {
        String requestId = NettyUtils.getRequestId(ctx.channel());
        RequestInput inputData = new RequestInput(requestId);
        CharSequence contentType = HttpUtil.getMimeType(req);
        if (HttpPostRequestDecoder.isMultipart(req)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)) {
            HttpDataFactory factory = new DefaultHttpDataFactory(6553500);
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(factory, req);
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
}
