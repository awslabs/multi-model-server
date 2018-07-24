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
package com.amazonaws.ml.mms;

import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelInferenceRequest;
import com.amazonaws.ml.mms.util.messages.ModelLoadModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.Predictions;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import com.amazonaws.ml.mms.util.messages.WorkerCommands;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.DelimiterBasedFrameDecoder;
import io.netty.handler.codec.Delimiters;
import io.netty.handler.codec.MessageToMessageDecoder;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import java.io.File;
import java.io.IOException;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MockWorker {

    private static final Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

    private EventLoopGroup bossGroup = NettyUtils.newEventLoopGroup(1);
    private EventLoopGroup workerGroup = NettyUtils.newEventLoopGroup(0);
    private ChannelFuture future;
    private AtomicBoolean stopped = new AtomicBoolean(true);

    public MockWorker() {}

    public static void main(String[] args) {
        try {
            new MockWorker().startAndWait();
        } catch (IOException | InterruptedException e) {
            logger.error("", e);
        }
    }

    public void startAndWait() throws InterruptedException, IOException {
        try {
            ChannelFuture f = start();

            logger.info("Mock worker started.");

            f.sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
            logger.info("Mock worker stopped.");
        }
        System.exit(-1);
    }

    public ChannelFuture start() throws InterruptedException, IOException {
        stopped.set(false);

        ServerBootstrap b = new ServerBootstrap();
        b.group(bossGroup, workerGroup).channel(NettyUtils.getServerUdsChannel());
        if (Boolean.getBoolean("DEBUG")) {
            b.handler(new LoggingHandler(LogLevel.INFO));
        }
        b.childHandler(
                new ChannelInitializer<Channel>() {

                    @Override
                    protected void initChannel(Channel ch) {
                        ChannelPipeline pipeline = ch.pipeline();
                        pipeline.addLast(
                                new DelimiterBasedFrameDecoder(
                                        81920000,
                                        Delimiters
                                                .lineDelimiter())); // TODO: Make this config based
                        pipeline.addLast(new StringDecoder());
                        pipeline.addLast(
                                new MessageToMessageDecoder<String>() {

                                    @Override
                                    protected void decode(
                                            ChannelHandlerContext ctx,
                                            String msg,
                                            List<Object> out) {
                                        JsonParser parser = new JsonParser();
                                        JsonObject json = (JsonObject) parser.parse(msg);
                                        String cmd = json.get("command").getAsString();
                                        if (WorkerCommands.LOAD
                                                .getCommand()
                                                .equalsIgnoreCase(cmd)) {
                                            out.add(
                                                    JsonUtils.GSON.fromJson(
                                                            json, ModelLoadModelRequest.class));
                                        } else {
                                            out.add(
                                                    JsonUtils.GSON.fromJson(
                                                            json, ModelInferenceRequest.class));
                                        }
                                    }
                                });
                        pipeline.addLast(new MockHandler());
                    }
                });

        SocketAddress address = NettyUtils.getSocketAddress(9000);
        if (address instanceof DomainSocketAddress) {
            File file = new File(((DomainSocketAddress) address).path());
            if (file.exists() && !file.delete()) {
                throw new IOException("Address already in use: " + file.getAbsolutePath());
            }
        }

        future = b.bind(address).sync();
        logger.info("Mock worker listening on: {}", address);

        return future.channel().closeFuture();
    }

    public boolean isRunning() {
        return !stopped.get();
    }

    public void stop() {
        if (stopped.get()) {
            return;
        }

        stopped.set(true);
        try {
            future.channel().close();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }

    private static final class MockHandler extends SimpleChannelInboundHandler<BaseModelRequest> {

        @Override
        public void channelActive(ChannelHandlerContext ctx) throws Exception {
            logger.debug("Mock worker channelActive");
            super.channelActive(ctx);
        }

        @Override
        public void channelRead0(ChannelHandlerContext ctx, BaseModelRequest msg) {
            logger.debug("Mock worker received: {}", msg.getModelName());
            ModelWorkerResponse resp = new ModelWorkerResponse();
            resp.setCode("200");
            resp.setMessage("Loaded.");
            if (msg instanceof ModelInferenceRequest) {
                List<RequestBatch> requestBatches = ((ModelInferenceRequest) msg).getRequestBatch();
                List<Predictions> predictions = new ArrayList<>(requestBatches.size());
                Base64.Encoder encoder = Base64.getEncoder();
                for (RequestBatch requestBatch : requestBatches) {
                    String requestId = requestBatch.getRequestId();
                    Predictions prediction = new Predictions();
                    prediction.setRequestId(requestId);
                    prediction.setValue(
                            encoder.encodeToString("OK".getBytes(StandardCharsets.UTF_8)));
                    predictions.add(prediction);
                }
                resp.setPredictions(predictions);
            }
            ByteBuf buf =
                    Unpooled.copiedBuffer(
                            JsonUtils.GSON.toJson(resp) + "\r\n", StandardCharsets.UTF_8);
            ctx.writeAndFlush(buf);

            //            Request<byte[]> request = new Request<>("application/json");
            //            request.setContent(msg.getPayloads().get(0).getData());
            //
            //            Noop noop = new Noop();
            //            noop.initialize(null);
            //
            //            List<Payload> payloads = msg.getPayloads();
            //
            //            Message result = new Message(modelName);
            //            for (Payload payload : payloads) {
            //                byte[] data = payload.getData();
            //
            //                Response<?> resp = noop.predict(request);
            //                String json = JsonUtils.GSON_PRETTY.toJson(resp.getContent());
            //                result.addPayload(
            //                        new Payload(payload.getId(), json.getBytes(StandardCharsets.UTF_8)));
            //            }
            //
            //            ctx.writeAndFlush(result);
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("", cause);
            NettyUtils.closeOnFlush(ctx.channel());
        }
    }
}
