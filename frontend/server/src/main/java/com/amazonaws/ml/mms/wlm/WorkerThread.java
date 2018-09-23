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
import com.amazonaws.ml.mms.util.codec.ModelRequestEncoder;
import com.amazonaws.ml.mms.util.codec.ModelResponseDecoder;
import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.InputParameter;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.RequestInput;
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
import java.io.IOException;
import java.net.SocketAddress;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerThread implements Runnable {

    static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);

    static final long WORKER_TIMEOUT = 2L;
    static final ModelRequestEncoder ENCODER = new ModelRequestEncoder();

    private ConfigManager configManager;
    private EventLoopGroup backendEventGroup;
    private int port;
    private Model model;

    private List<WorkerThread> parentThreads;
    private Channel backendChannel;
    private AtomicBoolean running = new AtomicBoolean(true);

    private BatchAggregator aggregator;
    private WorkerStateListener listener;
    ArrayBlockingQueue<ModelWorkerResponse> replies;
    private int gpuId;
    private long memory;
    private long startTime;
    private Thread currentThread;
    private String workerId;

    private WorkerState state;

    private WorkerLifeCycle lifeCycle;

    public WorkerThread(
            ConfigManager configManager,
            List<WorkerThread> parentThreads,
            EventLoopGroup backendEventGroup,
            int port,
            int gpuId,
            Model model,
            BatchAggregator aggregator,
            WorkerStateListener listener) {
        this.workerId = String.valueOf(port); // Unique across all workers.
        this.parentThreads = parentThreads;
        this.configManager = configManager;
        this.backendEventGroup = backendEventGroup;
        this.port = port;
        this.model = model;
        this.aggregator = aggregator;
        this.gpuId = gpuId;
        this.listener = listener;
        startTime = System.currentTimeMillis();
        lifeCycle = new WorkerLifeCycle(configManager, model);
        replies = new ArrayBlockingQueue<>(1);
    }

    @Override
    public void run() {
        currentThread = Thread.currentThread();
        currentThread.setName("BackendWorker-" + port);
        BaseModelRequest req = null;
        try {
            connect();

            while (running.get()) {
                req = aggregator.getRequest(workerId);

                backendChannel.write(req);
                backendChannel.flush();

                // TODO: Change this to configurable param
                long begin = System.currentTimeMillis();
                ModelWorkerResponse reply = replies.poll(WORKER_TIMEOUT, TimeUnit.MINUTES);

                long duration = System.currentTimeMillis() - begin;
                logger.info("Backend response time: {}", duration);

                if (reply != null) {
                    aggregator.sendResponse(reply);
                } else {
                    int val = model.incrFailedInfReqs();
                    logger.error("Number or consecutive unsuccessful inference {}", val);
                    throw new WorkerInitializationException(
                            "Backend worker did not respond in given time");
                }
                switch (req.getCommand()) {
                    case PREDICT:
                        model.resetFailedInfReqs();
                        break;
                    case LOAD:
                        if (reply.getCode() == 200) {
                            setState(WorkerState.WORKER_MODEL_LOADED);
                        } else {
                            setState(WorkerState.WORKER_ERROR);
                        }
                        break;
                    case UNLOAD:
                    case STATS:
                    default:
                        break;
                }
                req = null;
            }
        } catch (InterruptedException e) {
            logger.warn("Backend worker thread interrupted.");
        } catch (WorkerInitializationException e) {
            logger.error("Backend worker error", e);
        } catch (Throwable t) {
            logger.warn("Backend worker thread exception.", t);
        } finally {
            if (req != null) {
                aggregator.sendError(
                        req, ErrorCodes.INTERNAL_SERVER_ERROR_BACKEND_WORKER_INSTANTIATION);
            }
            setState(WorkerState.WORKER_STOPPED);
            lifeCycle.exit();
        }
    }

    public String getWorkerId() {
        return workerId;
    }

    public long getMemory() {
        return memory;
    }

    public void setMemory(long memory) {
        this.memory = memory;
    }

    private void connect() throws WorkerInitializationException {
        if (!configManager.isDebug() && !lifeCycle.startWorker(port)) {
            throw new WorkerInitializationException(
                    ErrorCodes.INTERNAL_SERVER_ERROR_BACKEND_WORKER_INSTANTIATION);
        }
        String modelName = model.getModelName();
        setState(WorkerState.WORKER_STARTED);
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
                                    p.addLast(ENCODER);
                                    p.addLast(new ModelResponseDecoder());
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
                                        logger.info("{} Worker disconnected.", getWorkerId());
                                        retry();
                                    });

            backendChannel
                    .newSucceededFuture()
                    .addListener(
                            (ChannelFutureListener)
                                    future -> {
                                        // TODO:
                                        // use gpu, batch size in load model command
                                        RequestInput input =
                                                new RequestInput(UUID.randomUUID().toString());
                                        if (gpuId >= 0) {
                                            input.addParameter(
                                                    new InputParameter(
                                                            "gpu", String.valueOf(gpuId)));
                                        }

                                        Job job =
                                                new Job(
                                                        null,
                                                        modelName,
                                                        WorkerCommands.LOAD,
                                                        input);
                                        model.addJob(workerId, job);
                                        latch.countDown();
                                    });

            if (!latch.await(WORKER_TIMEOUT, TimeUnit.MINUTES)) {
                throw new WorkerInitializationException(
                        ErrorCodes.INTERNAL_SERVER_ERROR_WORKER_HEALTH_CHECK_TIMEOUT,
                        "Worker failed to initialize within {} mins" + WORKER_TIMEOUT);
            }
        } catch (InterruptedException e) {
            throw new WorkerInitializationException(ErrorCodes.WORKER_INSTANTIATION_ERROR, e);
        } catch (Throwable t) {
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

    public int getPid() {
        return lifeCycle.getPid();
    }

    public void shutdown() {
        running.set(false);
        setState(WorkerState.WORKER_TERMINATED);
        if (backendChannel != null) {
            backendChannel.close();
        }
        if (currentThread != null) {
            currentThread.interrupt();
            aggregator.sendError(null, "Internal Failure");

            model.removeJobQueue(workerId);
        }
        // TODO: push current message back to queue, if no more worker,
        // drain the queue and send error back
    }

    void setState(WorkerState newState) {
        listener.notifyChangeState(model.getModelName(), newState);
        if (state != WorkerState.WORKER_TERMINATED) {
            this.state = newState;
        }
    }

    void retry() {
        if (state == WorkerState.WORKER_TERMINATED) {
            return;
        }

        logger.info("Retry worker: {}", workerId);
        ModelManager modelManager = ModelManager.getInstance();
        modelManager.submitTask(this);
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
