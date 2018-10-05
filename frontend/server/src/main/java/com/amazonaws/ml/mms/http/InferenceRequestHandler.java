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

import com.amazonaws.ml.mms.archive.ModelNotFoundException;
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
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class InferenceRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);

    /** Creates a new {@code InferenceRequestHandler} instance. */
    public InferenceRequestHandler() {}

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelNotFoundException {
        switch (segments[1]) {
            case "ping":
                ModelManager.getInstance().workerStatus(ctx);
                break;
            case "api-description":
                handleApiDescription(ctx);
                break;
            case "invocations":
                handleInvocations(ctx, req, decoder);
                break;
            case "predictions":
                handlePredictions(ctx, req, segments);
                break;
            default:
                handleLegacyPredict(ctx, req, decoder, segments);
                break;
        }
    }

    @Override
    protected void handleApiDescription(ChannelHandlerContext ctx) {
        NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listInferenceApis());
    }

    private void handlePredictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments)
            throws ModelNotFoundException {
        if (segments.length < 3) {
            throw new ResourceNotFoundException();
        }
        predict(ctx, req, null, segments[2]);
    }

    private void handleInvocations(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder)
            throws ModelNotFoundException {
        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        predict(ctx, req, decoder, modelName);
    }

    private void handleLegacyPredict(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelNotFoundException {
        if (segments.length < 3 || !"predict".equals(segments[2])) {
            throw new ResourceNotFoundException();
        }

        predict(ctx, req, decoder, segments[1]);
    }

    private void predict(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String modelName)
            throws ModelNotFoundException {
        RequestInput input = parseRequest(ctx, req, decoder);
        if (modelName == null) {
            modelName = input.getStringParameter("model_name");
            if (modelName == null) {
                throw new BadRequestException("Parameter model_name is required.");
            }
        }

        if (HttpMethod.OPTIONS.equals(req.method())) {
            ModelManager modelManager = ModelManager.getInstance();
            Model model = modelManager.getModels().get(modelName);
            if (model == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }

            String resp = OpenApiUtils.getModelApi(model);
            NettyUtils.sendJsonResponse(ctx, resp);
            return;
        }

        Job job = new Job(ctx, modelName, WorkerCommands.PREDICT, input);
        if (!ModelManager.getInstance().addJob(job)) {
            throw new ServiceUnavailableException(
                    "No worker is available to serve request: " + modelName);
        }
    }

    private static RequestInput parseRequest(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder) {
        String requestId = NettyUtils.getRequestId(ctx.channel());
        RequestInput inputData = new RequestInput(requestId);
        if (decoder != null) {
            for (Map.Entry<String, List<String>> entry : decoder.parameters().entrySet()) {
                String key = entry.getKey();
                for (String value : entry.getValue()) {
                    inputData.addParameter(new InputParameter(key, value));
                }
            }
        }

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
                logger.trace("End of multipart items.");
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
