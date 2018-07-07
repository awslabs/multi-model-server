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
package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.codec.MessageDecoder;
import com.amazonaws.ml.mms.util.codec.MessageEncoder;
import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.DelimiterBasedFrameDecoder;
import io.netty.handler.codec.Delimiters;
import io.netty.handler.codec.string.StringDecoder;
import java.io.IOException;
import java.net.SocketAddress;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerThread extends Thread {

    static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);

    //private static final StringDecoder DECODER = new StringDecoder();

    private ConfigManager configManager;
    private EventLoopGroup backendEventGroup;
    private int port;
    private Model model;

    private List<WorkerThread> parentThreads;
    private Channel backendChannel;
    private AtomicBoolean running = new AtomicBoolean(true);

    private BatchAggregator aggregator;
    ArrayBlockingQueue<ModelWorkerResponse> replies;
    private int gpuId;
    private Thread currentThread;

    private WorkerLifeCycle lifeCycle;

    public WorkerThread(
            ConfigManager configManager,
            List<WorkerThread> parentThreads,
            EventLoopGroup backendEventGroup,
            int port,
            int gpuId,
            Model model,
            BatchAggregator aggregator) {
        super("BackendWorker-" + port);
        this.parentThreads = parentThreads;
        this.configManager = configManager;
        this.backendEventGroup = backendEventGroup;
        this.port = port;
        this.model = model;
        this.aggregator = aggregator;
        this.gpuId = gpuId;
        lifeCycle = new WorkerLifeCycle(configManager);
        replies = new ArrayBlockingQueue<>(1);
        this.setDaemon(true);
    }

    @Override
    public void run() {
        currentThread = Thread.currentThread();
        BaseModelRequest req = null;
        try {
            while (running.get()) {
                req = aggregator.getRequest();
                backendChannel.write(req);
                backendChannel.flush();

                ModelWorkerResponse reply = replies.take();
                aggregator.sendResponse(reply);
                req = null;
            }
        } catch (InterruptedException e) {
            logger.warn("Backend worker thread interrupted.", e);
        } catch (Throwable t) {
            logger.warn("Backend worker thread exception.", t);
        } finally {
            if (req != null) {
                aggregator.sendError(req, "Worker execution error.");
            }
            lifeCycle.exit();
        }
    }

    public void connect() throws WorkerInitializationException {
        if (!configManager.isDebug() && !lifeCycle.startWorker(port)) {
            throw new WorkerInitializationException("Failed start worker process.");
        }
        final CountDownLatch latch = new CountDownLatch(1);

        try {
            Bootstrap b = new Bootstrap();
            b.group(backendEventGroup)
                    .channel(NettyUtils.getClientChannel())
                    .handler(
                            new ChannelInitializer<Channel>() {
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    p.addLast(
                                            new DelimiterBasedFrameDecoder(
                                                    81920000,
                                                    Delimiters
                                                            .lineDelimiter())); // TODO: Make this config based
                                    p.addLast(new StringDecoder());
                                    p.addLast(new MessageDecoder());
                                    p.addLast(new MessageEncoder());
                                    p.addLast(new WorkerHandler());
                                }
                            });

            SocketAddress address = NettyUtils.getSocketAddress(port);
            logger.debug("Connecting to: {}", address);
            backendChannel = b.connect(address).sync().channel();
            backendChannel
                    .closeFuture()
                    .addListener(
                            (ChannelFutureListener)
                                    future -> {
                                        latch.countDown();
                                        parentThreads.remove(WorkerThread.this); // NOPMD
                                        shutdown();
                                        logger.info("Worker disconnected.");
                                    });

            backendChannel
                    .newSucceededFuture()
                    .addListener((ChannelFutureListener) future -> latch.countDown());

            if (!sendLoadMessage(latch)) {
                lifeCycle.exit();
                throw new WorkerInitializationException("Failed to load the new model");
            }
        } catch (InterruptedException e) {
            lifeCycle.exit();
            throw new WorkerInitializationException(e);
        } catch (Throwable t) {
            lifeCycle.exit();

            // https://github.com/netty/netty/issues/2597
            if (t instanceof IOException) {
                throw new WorkerInitializationException(t);
            }
            throw t;
        }
    }

    public boolean sendLoadMessage(CountDownLatch latch) {
        int gpu = this.gpuId;
        try {
            latch.await();
            Job job = new Job(null, "load", new Payload(null, ""));
            model.addFirst(job);
        } catch (InterruptedException e) {
            logger.warn("Backend worker thread interrupted to start in gpu " + gpu, e);
            return false;
        } catch (Throwable t) {
            logger.warn("Backend worker thread exception.", t);
            return false;
        }

        return true;
    }

    public void shutdown() {
        running.set(false);
        backendChannel.close();
        if (currentThread != null) {
            currentThread.interrupt();
        }
        // TODO: push current message back to queue, if no more worker,
        // drain the queue and send error back
    }

    @ChannelHandler.Sharable
    private class WorkerHandler extends SimpleChannelInboundHandler<ModelWorkerResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, ModelWorkerResponse msg) {
            if (!replies.offer(msg)) {
                throw new IllegalStateException("Reply queue is full.");
            }
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("Unknown exception", cause);
            ctx.close();
        }
    }
}
