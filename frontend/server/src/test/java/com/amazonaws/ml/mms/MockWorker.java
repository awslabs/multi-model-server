package com.amazonaws.ml.mms;

import com.amazonaws.ml.mms.util.CodecUtils.MessageEncoder;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.wlm.Message;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import java.io.File;
import java.io.IOException;
import java.net.SocketAddress;
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
                        pipeline.addLast(new MessageEncoder());
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

    private static final class MockHandler extends SimpleChannelInboundHandler<Message> {

        @Override
        public void channelActive(ChannelHandlerContext ctx) throws Exception {
            logger.debug("Mock worker channelActive");
            super.channelActive(ctx);
        }

        @Override
        public void channelRead0(ChannelHandlerContext ctx, Message msg) {
            logger.debug("Mock worker received: {}", msg.getModelName());
            ctx.writeAndFlush(msg);
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("", cause);
            NettyUtils.closeOnFlush(ctx.channel());
        }
    }
}
