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
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, "Invalid HTTP request.");
            return;
        }

        try {
            QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
            String[] segments = decoder.path().split("/");

            switch (segments[1]) {
                case "ping":
                    ModelManager.getInstance().workerStatus(ctx);
                    break;
                case "api-description":
                    handleApiDescription(ctx);
                    break;
                default:
                    handleRequest(ctx, req, decoder, segments);
                    break;
            }
        } catch (IllegalArgumentException e) {
            logger.debug("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, e.getMessage());
        } catch (Throwable t) {
            logger.error("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR);
        }
    }

    protected abstract void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments);

    protected abstract void handleApiDescription(ChannelHandlerContext ctx);

    /** {@inheritDoc} */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        logger.error("", cause);
        ctx.close();
    }
}
