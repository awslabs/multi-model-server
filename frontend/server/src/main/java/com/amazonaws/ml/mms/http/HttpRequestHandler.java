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

import com.amazonaws.ml.mms.archive.ModelException;
import com.amazonaws.ml.mms.archive.ModelNotFoundException;
import com.amazonaws.ml.mms.servingsdk_impl.ModelServerContext;
import com.amazonaws.ml.mms.servingsdk_impl.ModelServerRequest;
import com.amazonaws.ml.mms.servingsdk_impl.ModelServerResponse;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.wlm.ModelManager;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.IOException;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;
import software.amazon.ai.mms.servingsdk.ModelServerEndpointException;

/**
 * A class handling inbound HTTP requests.
 *
 * <p>This class
 */
public abstract class HttpRequestHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    private static final Logger logger = LoggerFactory.getLogger(HttpRequestHandler.class);
    protected Map<String, ModelServerEndpoint> endpointMap;

    /** Creates a new {@code HttpRequestHandler} instance. */
    public HttpRequestHandler() {}

    /** {@inheritDoc} */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest req) {
        try {
            NettyUtils.requestReceived(ctx.channel(), req);
            if (!req.decoderResult().isSuccess()) {
                throw new BadRequestException("Invalid HTTP message.");
            }

            QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
            String path = decoder.path();
            if ("/".equals(path)) {
                if (HttpMethod.OPTIONS.equals(req.method())) {
                    handleApiDescription(ctx);
                    return;
                }
                throw new MethodNotAllowedException();
            }

            String[] segments = path.split("/");
            handleRequest(ctx, req, decoder, segments);
        } catch (ResourceNotFoundException | ModelNotFoundException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, e);
        } catch (BadRequestException | ModelException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, e);
        } catch (MethodNotAllowedException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.METHOD_NOT_ALLOWED, e);
        } catch (ServiceUnavailableException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.SERVICE_UNAVAILABLE, e);
        } catch (Throwable t) {
            logger.error("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, t);
        }
    }

    private void run(
            ModelServerEndpoint endpoint,
            FullHttpRequest req,
            FullHttpResponse rsp,
            QueryStringDecoder decoder,
            String method)
            throws IOException {
        switch (method) {
            case "GET":
                endpoint.doGet(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            case "PUT":
                endpoint.doPut(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            case "DELETE":
                endpoint.doDelete(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            case "POST":
                endpoint.doPost(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            default:
                throw new ServiceUnavailableException("Invalid HTTP method received");
        }
    }

    protected void handleCustomEndpoint(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            String[] segments,
            QueryStringDecoder decoder) {
        ModelServerEndpoint endpoint = endpointMap.get(segments[1]);
        Runnable r =
                () -> {
                    Long start = System.currentTimeMillis();
                    FullHttpResponse rsp =
                            new DefaultFullHttpResponse(
                                    HttpVersion.HTTP_1_1, HttpResponseStatus.OK, false);
                    try {
                        run(endpoint, req, rsp, decoder, req.method().toString());
                        NettyUtils.sendHttpResponse(ctx, rsp, true);
                        logger.info(
                                "Running \"{}\" endpoint took {} ms",
                                segments[0],
                                System.currentTimeMillis() - start);
                    } catch (ModelServerEndpointException me) {
                        NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, me);
                        logger.error("Error thrown by the model endpoint plugin.", me);
                    } catch (IOException ioe) {
                        NettyUtils.sendError(
                                ctx,
                                HttpResponseStatus.INTERNAL_SERVER_ERROR,
                                ioe,
                                "I/O error while running the custom endpoint");
                        logger.error("I/O error while running the custom endpoint.", ioe);
                    } catch (Throwable e) {
                        NettyUtils.sendError(
                                ctx,
                                HttpResponseStatus.INTERNAL_SERVER_ERROR,
                                e,
                                "Unknown exception");
                        logger.error("Unknown exception", e);
                    }
                };
        ModelManager.getInstance().submitTask(r);
    }

    protected abstract void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException;

    protected abstract void handleApiDescription(ChannelHandlerContext ctx);

    /** {@inheritDoc} */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        logger.error("", cause);
        ctx.close();
    }
}
