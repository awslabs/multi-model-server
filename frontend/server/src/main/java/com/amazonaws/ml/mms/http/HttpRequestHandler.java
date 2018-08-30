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
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.wlm.ModelManager;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests.
 *
 * <p>This class
 */
public abstract class HttpRequestHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

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
            handleGeneralRequest(ctx, req);
        } catch (IllegalArgumentException e) {
            logger.debug("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, e.getMessage());
        } catch (Throwable t) {
            logger.error("", t);
            NettyUtils.sendError(
                    ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, ErrorCodes.UNKNOWN_ERROR);
        }
    }

    protected boolean handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        return false;
    }

    protected abstract void handleApiDescription(ChannelHandlerContext ctx);

    protected void handleGeneralRequest(ChannelHandlerContext ctx, FullHttpRequest req) {
        QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
        String[] segments = decoder.path().split("/");

        if (!handleRequest(ctx, req, decoder, segments)) {
            switch (segments[1]) {
                case "ping":
                    handlePing(ctx);
                    break;
                case "api-description":
                    handleApiDescription(ctx);
                    break;
                default:
                    NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, ErrorCodes.INVALID_URI);
                    break;
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        logger.error("", cause);
        ctx.close();
    }

    private void handlePing(ChannelHandlerContext ctx) {
        ModelManager.getInstance().workerStatus(ctx);
    }
}
