package com.amazonaws.ml.mms;

import com.amazonaws.ml.mms.archive.InvalidModelException;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.wlm.MessageCodec;
import com.amazonaws.ml.mms.wlm.WorkerInitializationException;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class ModelServerTest {

    private static final Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

    private ConfigManager configManager;
    private ModelServer server;
    private MockWorker worker;
    CountDownLatch latch;
    String result;
    private String openApiResult;

    @BeforeSuite
    public void beforeSuite()
            throws InterruptedException, InvalidModelException, WorkerInitializationException,
                    IOException {
        System.setProperty("DEBUG", "true");
        configManager = new ConfigManager();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);

        worker = new MockWorker();
        worker.start();

        server = new ModelServer(configManager);
        server.initModelStore();
        server.start();

        try (InputStream is = new FileInputStream("src/test/resources/open_api.txt")) {
            openApiResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
        }
    }

    @AfterSuite
    public void afterSuite() {
        server.stop();
        worker.stop();
    }

    @Test
    public void test() throws InterruptedException {
        Channel channel = null;
        for (int i = 0; i < 5; ++i) {
            channel = connect();
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }
        Assert.assertNotNull(channel, "Model Server should have started.");

        testRoot(channel);
        testPing(channel);
        testApiDescription(channel);
        testUnregisterModel(channel);
        testLoadModel(channel);
        testScaleModel(channel);
        testInvocations(channel);
        channel.close();
    }

    private void testRoot(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testPing(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "{\"status\":\"healthy\"}");
    }

    private void testApiDescription(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/api-description");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, openApiResult);
    }

    private void testLoadModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/register?url=noop-v0.1.model");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "{\"status\":\"Model registered\"}");
    }

    private void testScaleModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.GET,
                        "/scale?model_name=noop_v0.1&min_worker=1");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "{\"status\":\"Worker updated\"}");
    }

    private void testUnregisterModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/unregister?model_name=noop_v0.1");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "{\"status\":\"Model unregistered\"}");
    }

    private void testInvocations(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.GET,
                        "/invocations?model_name=noop_v0.1&data=test");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "test");
    }

    private Channel connect() {
        try {
            Bootstrap b = new Bootstrap();
            b.group(new NioEventLoopGroup(1))
                    .channel(NioSocketChannel.class)
                    .handler(
                            new ChannelInitializer<Channel>() {
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    p.addLast(new HttpClientCodec());
                                    p.addLast(new HttpContentDecompressor());
                                    p.addLast(new HttpObjectAggregator(6553600));
                                    p.addLast(new MessageCodec());
                                    p.addLast(new TestHandler());
                                }
                            });

            SocketAddress address = new InetSocketAddress("127.0.0.1", configManager.getPort());
            return b.connect(address).sync().channel();
        } catch (Throwable t) {
            logger.warn("Connect error.", t);
        }
        return null;
    }

    @ChannelHandler.Sharable
    private class TestHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            result = msg.content().toString(StandardCharsets.UTF_8);
            latch.countDown();
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("Unknown exception", cause);
            ctx.close();
        }
    }
}
