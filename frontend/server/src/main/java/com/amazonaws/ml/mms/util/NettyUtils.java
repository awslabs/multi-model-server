package com.amazonaws.ml.mms.util;

import com.amazonaws.ml.mms.http.ErrorResponse;
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
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.Attribute;
import io.netty.handler.codec.http.multipart.FileUpload;
import io.netty.handler.codec.http.multipart.InterfaceHttpData;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;

/** A utility class that handling Netty request and response. */
public final class NettyUtils {

    private static final String UDS_PREFIX = "/tmp/.mms.worker.";

    private NettyUtils() {}

    public static void sendJsonResponse(ChannelHandlerContext ctx, Object json) {
        sendJsonResponse(ctx, JsonUtils.GSON_PRETTY.toJson(json));
    }

    public static void sendJsonResponse(ChannelHandlerContext ctx, String json) {
        FullHttpResponse resp =
                new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK);
        resp.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        resp.content().writeCharSequence(json, CharsetUtil.UTF_8);
        sendHttpResponse(ctx, resp, true);
    }

    /**
     * Send simple HTTP response to client and close connection.
     *
     * @param ctx ChannelHandlerContext
     * @param status HttpResponseStatus to send
     */
    public static void sendError(ChannelHandlerContext ctx, HttpResponseStatus status) {
        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status);
        ErrorResponse error =
                new ErrorResponse(status.code(), status.toString(), status.reasonPhrase());
        resp.content().writeCharSequence(JsonUtils.GSON.toJson(error), CharsetUtil.UTF_8);
        sendHttpResponse(ctx, resp, false);
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
        HttpUtil.setContentLength(resp, resp.content().readableBytes());
        if (!keepAlive || resp.status().code() >= 400) {
            ChannelFuture f = ctx.channel().writeAndFlush(resp);
            f.addListener(ChannelFutureListener.CLOSE);
        } else {
            resp.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
            ctx.channel().writeAndFlush(resp);
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
            return new DomainSocketAddress(UDS_PREFIX + port);
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
        return Integer.parseInt(value);
    }

    public static String getFormField(InterfaceHttpData data) {
        if (data == null) {
            return null;
        }

        InterfaceHttpData.HttpDataType type = data.getHttpDataType();
        if (type != InterfaceHttpData.HttpDataType.Attribute) {
            throw new IllegalArgumentException("Except form field, but got " + type);
        }

        Attribute attribute = (Attribute) data;
        try {
            return attribute.getValue();
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    public static byte[] getFormData(InterfaceHttpData data) {
        if (data == null) {
            return null;
        }

        switch (data.getHttpDataType()) {
            case Attribute:
                Attribute attribute = (Attribute) data;
                try {
                    return attribute.getValue().getBytes(StandardCharsets.UTF_8);
                } catch (IOException e) {
                    throw new AssertionError(e);
                }
            case FileUpload:
                FileUpload fileUpload = (FileUpload) data;
                try {
                    return getBytes(fileUpload.getByteBuf());
                } catch (IOException e) {
                    throw new AssertionError(e);
                }
            default:
                throw new IllegalArgumentException(
                        "Except form field, but got " + data.getHttpDataType());
        }
    }
}
