package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.util.ConfigManager;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.unix.DomainSocketAddress;
import java.io.IOException;
import java.net.SocketAddress;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerThread extends Thread {

    static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);

    private static final String UDS_PREFIX = "/tmp/.mms.worker.";

    private ConfigManager configManager;
    private EventLoopGroup backendEventGroup;
    private int port;
    private Model model;

    private List<WorkerThread> parentThreads;
    private Channel backendChannel;
    private AtomicBoolean running = new AtomicBoolean(true);

    private BatchAggregator aggregator;
    ArrayBlockingQueue<Message> replies;

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
        lifeCycle = new WorkerLifeCycle(configManager, gpuId);
        replies = new ArrayBlockingQueue<>(1);
        this.setDaemon(true);
    }

    @Override
    public void run() {
        Message message = null;
        try {
            while (running.get()) {
                message = aggregator.getRequest();
                backendChannel.write(message);
                backendChannel.flush();

                Message reply = replies.take();
                aggregator.sendResponse(reply);
                message = null;
            }
        } catch (InterruptedException e) {
            logger.warn("Backend worker thread interrupted.", e);
        } catch (Throwable t) {
            logger.warn("Backend worker thread exception.", t);
        } finally {
            if (message != null) {
                aggregator.sendError(message, "Worker execution error.");
            }
            lifeCycle.exit();
        }
    }

    public void connect() throws WorkerInitializationException {
        if (!configManager.isDebug() && !lifeCycle.startWorker(port, model)) {
            throw new WorkerInitializationException("Failed start worker process.");
        }

        try {
            Bootstrap b = new Bootstrap();
            b.group(backendEventGroup)
                    .channel(configManager.getChannel())
                    .handler(
                            new ChannelInitializer<Channel>() {
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    p.addLast(new MessageCodec());
                                    p.addLast(new MxNetHandler());
                                }
                            });

            SocketAddress address = new DomainSocketAddress(UDS_PREFIX + port);
            backendChannel = b.connect(address).sync().channel();
            backendChannel
                    .closeFuture()
                    .addListener(
                            (ChannelFutureListener)
                                    future -> {
                                        parentThreads.remove(WorkerThread.this); // NOPMD
                                        shutdown();
                                        logger.info("Worker disconnection.");
                                    });
        } catch (InterruptedException e) {
            throw new WorkerInitializationException(e);
        } catch (Throwable t) {
            // https://github.com/netty/netty/issues/2597
            if (t instanceof IOException) {
                throw new WorkerInitializationException(t);
            }
            throw t;
        }
    }

    public void shutdown() {
        running.set(false);
        backendChannel.close();
        interrupt();
        // TODO: push current message back to queue, if no more worker,
        // drain the queue and send error back
    }

    @ChannelHandler.Sharable
    private class MxNetHandler extends SimpleChannelInboundHandler<Message> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, Message msg) {
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
