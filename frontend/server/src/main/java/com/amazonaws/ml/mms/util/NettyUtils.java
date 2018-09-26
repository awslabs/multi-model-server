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
package com.amazonaws.ml.mms.util;

import com.amazonaws.ml.mms.http.ErrorResponse;
import com.amazonaws.ml.mms.http.Session;
import com.amazonaws.ml.mms.util.messages.InputParameter;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.epoll.EpollServerDomainSocketChannel;
import io.netty.channel.epoll.EpollServerSocketChannel;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.kqueue.KQueueServerDomainSocketChannel;
import io.netty.channel.kqueue.KQueueServerSocketChannel;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.Attribute;
import io.netty.handler.codec.http.multipart.FileUpload;
import io.netty.handler.codec.http.multipart.InterfaceHttpData;
import io.netty.util.AttributeKey;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A utility class that handling Netty request and response. */
public final class NettyUtils {

    private static final Logger logger = LoggerFactory.getLogger("ACCESS_LOG");

    private static final String REQUEST_ID = "x-request-id";
    private static final AttributeKey<Session> SESSION_KEY = AttributeKey.valueOf("session");

    private NettyUtils() {}

    public static void requestReceived(Channel channel, HttpRequest request) {
        Session session = channel.attr(SESSION_KEY).get();
        assert session == null;

        String remoteIp = channel.remoteAddress().toString();
        channel.attr(SESSION_KEY).set(new Session(remoteIp, request));
    }

    public static String getRequestId(Channel channel) {
        Session accessLog = channel.attr(SESSION_KEY).get();
        if (accessLog != null) {
            return accessLog.getRequestId();
        }
        return null;
    }

    public static void sendJsonResponse(ChannelHandlerContext ctx, Object json) {
        sendJsonResponse(ctx, JsonUtils.GSON_PRETTY.toJson(json), HttpResponseStatus.OK);
    }

    public static void sendJsonResponse(
            ChannelHandlerContext ctx, Object json, HttpResponseStatus status) {
        sendJsonResponse(ctx, JsonUtils.GSON_PRETTY.toJson(json), status);
    }

    public static void sendJsonResponse(ChannelHandlerContext ctx, String json) {
        sendJsonResponse(ctx, json, HttpResponseStatus.OK);
    }

    public static void sendJsonResponse(
            ChannelHandlerContext ctx, String json, HttpResponseStatus status) {
        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status);
        resp.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        ByteBuf content = resp.content();
        content.writeCharSequence(json, CharsetUtil.UTF_8);
        content.writeByte('\n');
        sendHttpResponse(ctx, resp, true);
    }

    /**
     * Send simple HTTP response to client and close connection.
     *
     * @param ctx ChannelHandlerContext
     * @param status HttpResponseStatus to send
     */
    public static void sendError(ChannelHandlerContext ctx, HttpResponseStatus status) {
        sendError(ctx, status, status.reasonPhrase());
    }

    public static void sendError(
            ChannelHandlerContext ctx, HttpResponseStatus status, String errorMessage) {
        ErrorResponse error =
                new ErrorResponse(status.code(), status.codeAsText().toString(), errorMessage);
        sendJsonResponse(ctx, error, status);
    }

    /**
     * Send HTTP response to client.
     *
     * @param ctx ChannelHandlerContext
     * @param resp HttpResponse to send
     * @param keepAlive if keep the connection
     */
    public static void sendHttpResponse(
            ChannelHandlerContext ctx, FullHttpResponse resp, boolean keepAlive) {
        // Send the response and close the connection if necessary.
        Channel channel = ctx.channel();
        Session session = channel.attr(SESSION_KEY).getAndSet(null);
        if (session != null) {
            // session might be recycled if channel is closed already.
            session.setCode(resp.status().code());
            resp.headers().set(REQUEST_ID, session.getRequestId());
            logger.info(session.toString());
        }
        HttpUtil.setContentLength(resp, resp.content().readableBytes());
        if (!keepAlive || resp.status().code() >= 400) {
            ChannelFuture f = channel.writeAndFlush(resp);
            f.addListener(ChannelFutureListener.CLOSE);
        } else {
            resp.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
            channel.writeAndFlush(resp);
        }
    }

    /** Closes the specified channel after all queued write requests are flushed. */
    public static void closeOnFlush(Channel ch) {
        if (ch.isActive()) {
            ch.writeAndFlush(Unpooled.EMPTY_BUFFER).addListener(ChannelFutureListener.CLOSE);
        }
    }

    public static EventLoopGroup newEventLoopGroup(int threads) {
        if (Epoll.isAvailable()) {
            return new EpollEventLoopGroup(threads);
        } else if (KQueue.isAvailable()) {
            return new KQueueEventLoopGroup(threads);
        }

        return new NioEventLoopGroup(threads);
    }

    public static Class<? extends ServerChannel> getServerChannel() {
        if (Epoll.isAvailable()) {
            return EpollServerSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            return KQueueServerSocketChannel.class;
        }

        return NioServerSocketChannel.class;
    }

    public static Class<? extends ServerChannel> getServerUdsChannel() {
        if (Epoll.isAvailable()) {
            return EpollServerDomainSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            return KQueueServerDomainSocketChannel.class;
        }

        return NioServerSocketChannel.class;
    }

    public static Class<? extends Channel> getClientChannel() {
        if (Epoll.isAvailable()) {
            return EpollDomainSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            return KQueueDomainSocketChannel.class;
        }

        return NioSocketChannel.class;
    }

    public static SocketAddress getSocketAddress(int port) {
        if (Epoll.isAvailable() || KQueue.isAvailable()) {
            String uds = System.getProperty("java.io.tmpdir") + "/.mms.sock." + port;
            return new DomainSocketAddress(uds);
        }
        return new InetSocketAddress("127.0.0.1", port);
    }

    public static byte[] getBytes(ByteBuf buf) {
        if (buf.hasArray()) {
            return buf.array();
        }

        byte[] ret = new byte[buf.readableBytes()];
        int readerIndex = buf.readerIndex();
        buf.getBytes(readerIndex, ret);
        return ret;
    }

    public static String getParameter(QueryStringDecoder decoder, String key, String def) {
        List<String> param = decoder.parameters().get(key);
        if (param != null && !param.isEmpty()) {
            return param.get(0);
        }
        return def;
    }

    public static int getIntParameter(QueryStringDecoder decoder, String key, int def) {
        String value = getParameter(decoder, key, null);
        if (value == null) {
            return def;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return def;
        }
    }

    public static InputParameter getFormData(InterfaceHttpData data) {
        if (data == null) {
            return null;
        }

        String name = data.getName();
        switch (data.getHttpDataType()) {
            case Attribute:
                Attribute attribute = (Attribute) data;
                try {
                    return new InputParameter(name, attribute.getValue());
                } catch (IOException e) {
                    throw new AssertionError(e);
                }
            case FileUpload:
                FileUpload fileUpload = (FileUpload) data;
                String contentType = fileUpload.getContentType();
                try {
                    return new InputParameter(name, getBytes(fileUpload.getByteBuf()), contentType);
                } catch (IOException e) {
                    throw new AssertionError(e);
                }
            default:
                throw new IllegalArgumentException(
                        "Except form field, but got " + data.getHttpDataType());
        }
    }
}
