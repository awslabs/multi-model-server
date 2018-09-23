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
package com.amazonaws.ml.mms.cts;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.handler.timeout.ReadTimeoutException;
import io.netty.handler.timeout.ReadTimeoutHandler;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HttpClient {

    static final Logger logger = LoggerFactory.getLogger(HttpClient.class);

    private int managementPort;
    private int inferencePort;

    private Bootstrap bootstrap;
    private ClientHandler handler;

    public HttpClient(int managementPort, int inferencePort) {
        this.managementPort = managementPort;
        this.inferencePort = inferencePort;
        handler = new ClientHandler();
        bootstrap = bootstrap(handler);
    }

    public boolean registerModel(String modelName, String modelUrl)
            throws InterruptedException, IOException {
        Channel channel = connect(bootstrap, managementPort);

        String uri =
                "/models?url="
                        + URLEncoder.encode(modelUrl, StandardCharsets.UTF_8.name())
                        + "&model_name="
                        + modelName
                        + "&initial_workers=1&synchronous=true";

        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, uri);
        channel.writeAndFlush(req).sync();

        channel.closeFuture().sync();

        int statusCode = handler.getStatusCode();
        String ret = handler.getContent();
        if (statusCode == 200) {
            logger.info("registerModel: {} success.", modelName);
            logger.trace(ret);
            return true;
        }
        logger.warn("registerModel: {} failed: {}", modelUrl, ret);
        return false;
    }

    public boolean unregisterModel(String modelName) throws InterruptedException, IOException {
        Channel channel = connect(bootstrap, managementPort);

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.DELETE,
                        "/models/" + URLEncoder.encode(modelName, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req).sync();

        channel.closeFuture().sync();

        int statusCode = handler.getStatusCode();
        String ret = handler.getContent();
        if (statusCode == 200) {
            logger.info("unregisterModel: {} success.", modelName);
            logger.trace(ret);
            return true;
        }
        logger.warn("unregisterModel: {} failed: {}", modelName, ret);
        return false;
    }

    public boolean predict(String modelName, byte[] content, CharSequence contentType)
            throws InterruptedException, IOException {
        Channel channel = connect(bootstrap, inferencePort);

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/predictions/"
                                + URLEncoder.encode(modelName, StandardCharsets.UTF_8.name()));
        req.content().writeBytes(content);
        HttpUtil.setContentLength(req, content.length);
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, contentType);
        channel.writeAndFlush(req).sync();

        channel.closeFuture().sync();

        int statusCode = handler.getStatusCode();
        String ret = handler.getContent();
        if (statusCode == 200) {
            logger.info("predict: {} success.", modelName);
            logger.trace(ret);
            return true;
        }
        logger.warn("predict: {} failed: {}", modelName, ret);
        return false;
    }

    public boolean predict(
            String modelName, DefaultFullHttpRequest req, HttpPostRequestEncoder requestEncoder)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        Channel channel = connect(bootstrap, inferencePort);

        req.setUri("/predictions/" + URLEncoder.encode(modelName, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(requestEncoder.finalizeRequest());
        if (requestEncoder.isChunked()) {
            channel.writeAndFlush(requestEncoder).sync();
        }

        channel.closeFuture().sync();

        int statusCode = handler.getStatusCode();
        String ret = handler.getContent();
        if (statusCode == 200) {
            logger.info("predict: {} success.", modelName);
            logger.trace(ret);
            return true;
        }
        logger.warn("predict: {} failed: {}", modelName, ret);
        return false;
    }

    private Bootstrap bootstrap(ClientHandler handler) {
        Bootstrap b = new Bootstrap();
        b.group(new NioEventLoopGroup(1))
                .channel(NioSocketChannel.class)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10 * 1000)
                .handler(
                        new ChannelInitializer<Channel>() {
                            @Override
                            public void initChannel(Channel ch) {
                                ChannelPipeline p = ch.pipeline();
                                p.addLast(new ReadTimeoutHandler(10 * 60 * 1000));
                                p.addLast(new HttpClientCodec());
                                p.addLast(new HttpContentDecompressor());
                                p.addLast(new ChunkedWriteHandler());
                                p.addLast(new HttpObjectAggregator(6553600));
                                p.addLast(handler);
                            }
                        });
        return b;
    }

    private Channel connect(Bootstrap b, int port) throws InterruptedException {
        SocketAddress address = new InetSocketAddress("127.0.0.1", port);
        return b.connect(address).sync().channel();
    }

    @ChannelHandler.Sharable
    private static final class ClientHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        private int statusCode;
        private String content;

        public ClientHandler() {}

        public int getStatusCode() {
            return statusCode;
        }

        public String getContent() {
            return content;
        }

        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            statusCode = msg.status().code();
            content = msg.content().toString(StandardCharsets.UTF_8);
            ctx.close();
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            if (cause instanceof IOException) {
                content = "Failed to connect to MMS";
            } else if (cause instanceof ReadTimeoutException) {
                content = "Request to MMS timeout.";
            } else {
                content = cause.getMessage();
                if (content == null) {
                    content = "NullPointException";
                }
                logger.error("Unknown exception", cause);
            }
            statusCode = 500;
            ctx.close();
        }
    }
}
