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

import com.amazonaws.ml.mms.common.ErrorCodes;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.codec.MessageDecoder;
import com.amazonaws.ml.mms.util.codec.MessageEncoder;
import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelInputs;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import com.amazonaws.ml.mms.util.messages.WorkerCommands;
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
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerThread extends Thread {

    static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);
    static final long WORKER_TIMEOUT = 2L;
    static final StringDecoder STRING_DECODER = new StringDecoder();
    static final MessageDecoder MSG_DECODER = new MessageDecoder();
    static final MessageEncoder MSG_ENCODER = new MessageEncoder();

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
    private long startTime;
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
        super("BackendWorker-" + port); // ** IMPORTANT NOTE**: THIS NAME SHOULD BE UNIQUE..
        this.parentThreads = parentThreads;
        this.configManager = configManager;
        this.backendEventGroup = backendEventGroup;
        this.port = port;
        this.model = model;
        this.aggregator = aggregator;
        this.gpuId = gpuId;
        startTime = System.currentTimeMillis();
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
                req = aggregator.getRequest(getName());
                backendChannel.write(req);
                backendChannel.flush();

                // TODO: Change this to configurable param
                ModelWorkerResponse reply = replies.poll(WORKER_TIMEOUT, TimeUnit.MINUTES);

                if (reply != null) {
                    aggregator.sendResponse(reply);
                } else {
                    throw new WorkerInitializationException(
                            "Backend worker did not respond in given time");
                }

                if (req.getCommand().equals(WorkerCommands.PREDICT)) {
                    model.resetFailedInfReqs();
                }

                req = null;
            }
        } catch (InterruptedException e) {
            logger.warn("Backend worker thread interrupted.", e);
        } catch (WorkerInitializationException wi) {
            logger.error("Backend worker error", wi);
        } catch (Throwable t) {
            logger.warn("Backend worker thread exception.", t);
        } finally {
            if (req != null) {
                aggregator.sendError(
                        req, ErrorCodes.INTERNAL_SERVER_ERROR_BACKEND_WORKER_INSTANTIATION);
            }
            lifeCycle.exit();
        }
    }

    public void connect() throws WorkerInitializationException {
        if (!configManager.isDebug() && !lifeCycle.startWorker(port)) {
            throw new WorkerInitializationException(
                    ErrorCodes.INTERNAL_SERVER_ERROR_BACKEND_WORKER_INSTANTIATION);
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
                                    p.addLast(STRING_DECODER);
                                    p.addLast(MSG_DECODER);
                                    p.addLast(MSG_ENCODER);
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
                                        logger.info("{} Worker disconnected.", getName());
                                    });

            backendChannel
                    .newSucceededFuture()
                    .addListener(
                            (ChannelFutureListener)
                                    future -> {
                                        // TODO:
                                        // use gpu, batch size in load model command
                                        RequestBatch input = new RequestBatch();
                                        if (gpuId > 0) {
                                            input.addModelInput(
                                                    new ModelInputs("gpu", String.valueOf(gpuId)));
                                        }

                                        Job job =
                                                new Job(
                                                        null,
                                                        model.getModelName(),
                                                        WorkerCommands.LOAD,
                                                        input);
                                        model.addJob(getName(), job);
                                        latch.countDown();
                                    });

            if (!latch.await(WORKER_TIMEOUT, TimeUnit.MINUTES)) {
                throw new WorkerInitializationException(
                        ErrorCodes.INTERNAL_SERVER_ERROR_WORKER_HEALTH_CHECK_TIMEOUT,
                        "Worker failed to initialize within {} mins" + WORKER_TIMEOUT);
            }

        } catch (InterruptedException e) {
            lifeCycle.exit();
            throw new WorkerInitializationException(ErrorCodes.WORKER_INSTANTIATION_ERROR, e);
        } catch (Throwable t) {
            lifeCycle.exit();

            // https://github.com/netty/netty/issues/2597
            if (t instanceof IOException) {
                throw new WorkerInitializationException(
                        ErrorCodes.INTERNAL_SERVER_ERROR_WORKER_LISTEN_FAILURE, t);
            }
            throw t;
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    public int getGpuId() {
        return gpuId;
    }

    public long getStartTime() {
        return startTime;
    }

    public void shutdown() {
        running.set(false);
        backendChannel.close();
        ModelManager manager = ModelManager.getInstance();
        if (currentThread != null) {
            currentThread.interrupt();
            try {
                if (parentThreads.size() < model.getMinWorkers()) {
                    // minWorkers is the numWorkers configured for this model. If parent thread group
                    // has less threads than expected minWorkers, we should restart the workers.
                    manager.updateModel(
                            model.getModelName(), model.getMinWorkers(), model.getMinWorkers());
                }
                aggregator.sendError(null, "Internal Failure");
            } catch (WorkerInitializationException wie) {
                logger.error(
                        "Error restarting the backend worker for model name {}",
                        model.getModelName());
            }

            model.removeJobQueue(getName());
            int val = model.incrFailedInfReqs();
            logger.error("Number or consecutive unsuccessful inference {}", val);
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
