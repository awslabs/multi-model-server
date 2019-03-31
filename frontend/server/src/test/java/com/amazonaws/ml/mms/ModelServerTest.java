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

import com.amazonaws.ml.mms.http.DescribeModelResponse;
import com.amazonaws.ml.mms.http.ErrorResponse;
import com.amazonaws.ml.mms.http.ListModelsResponse;
import com.amazonaws.ml.mms.http.StatusResponse;
import com.amazonaws.ml.mms.metrics.Dimension;
import com.amazonaws.ml.mms.metrics.Metric;
import com.amazonaws.ml.mms.metrics.MetricManager;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.Connector;
import com.amazonaws.ml.mms.util.JsonUtils;
import com.google.gson.JsonParseException;
import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.codec.http.multipart.MemoryFileUpload;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.CharsetUtil;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class ModelServerTest {

    private static final String ERROR_NOT_FOUND =
            "Requested resource is not found, please refer to API document.";
    private static final String ERROR_METHOD_NOT_ALLOWED =
            "Requested method is not allowed, please refer to API document.";

    private ConfigManager configManager;
    private ModelServer server;
    CountDownLatch latch;
    HttpResponseStatus httpStatus;
    String result;
    HttpHeaders headers;
    private String listInferenceApisResult;
    private String listManagementApisResult;
    private String noopApiResult;

    static {
        TestUtils.init();
    }

    @BeforeSuite
    public void beforeSuite() throws InterruptedException, IOException, GeneralSecurityException {
        ConfigManager.init(new ConfigManager.Arguments());
        configManager = ConfigManager.getInstance();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);

        server = new ModelServer(configManager);
        server.start();

        try (InputStream is = new FileInputStream("src/test/resources/inference_open_api.json")) {
            listInferenceApisResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
        }

        try (InputStream is = new FileInputStream("src/test/resources/management_open_api.json")) {
            listManagementApisResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
        }

        try (InputStream is = new FileInputStream("src/test/resources/describe_api.json")) {
            noopApiResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
        }
    }

    @AfterSuite
    public void afterSuite() {
        server.stop();
    }

    @Test
    public void test()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        Channel channel = null;
        Channel managementChannel = null;
        for (int i = 0; i < 5; ++i) {
            channel = connect(false);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }

        for (int i = 0; i < 5; ++i) {
            managementChannel = connect(true);
            if (managementChannel != null) {
                break;
            }
            Thread.sleep(100);
        }

        Assert.assertNotNull(channel, "Failed to connect to inference port.");
        Assert.assertNotNull(managementChannel, "Failed to connect to management port.");

        testPing(channel);

        testRoot(channel, listInferenceApisResult);
        testRoot(managementChannel, listManagementApisResult);
        testApiDescription(channel, listInferenceApisResult);

        testDescribeApi(channel);
        testUnregisterModel(managementChannel);
        testLoadModel(managementChannel);
        testSyncScaleModel(managementChannel);
        testScaleModel(managementChannel);
        testListModels(managementChannel);
        testDescribeModel(managementChannel);

        testInitialModelDefaultBatchSettings(managementChannel);
        testLoadModelWithBatchSizeOverride(managementChannel);
        testLoadModelWithInitialWorkers(managementChannel);
        testLoadModelWithBatchSize(managementChannel);

        testPredictions(channel);
        testPredictionsBinary(channel);
        testPredictionsJson(channel);
        testInvocationsJson(channel);
        testInvocationsMultipart(channel);
        testLegacyPredict(channel);
        testPredictionsInvalidRequestSize(channel);
        testPredictionsValidRequestSize(channel);
        testMetricManager();

        channel.close();
        managementChannel.close();

        // negative test case, channel will be closed by server
        testInvalidRootRequest();
        testInvalidInferenceUri();
        testInvalidPredictionsUri();
        testInvalidDescribeModel();
        testPredictionsModelNotFound();

        testInvalidManagementUri();
        testInvalidModelsMethod();
        testInvalidModelMethod();
        testDescribeModelNotFound();
        testRegisterModelMissingUrl();
        testRegisterModelInvalidRuntime();
        testRegisterModelNotFound();
        testRegisterModelMalformedUrl();
        testRegisterModelConnectionFailed();
        testRegisterModelHttpError();
        testRegisterModelInvalidPath();
        testScaleModelNotFound();
        testScaleModelFailure();
        testUnregisterModelNotFound();
        testInvalidModel();
    }

    private void testRoot(Channel channel, String expected) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/");
        channel.writeAndFlush(req).sync();
        latch.await();

        Assert.assertEquals(result, expected);
    }

    private void testPing(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Healthy");
        Assert.assertTrue(headers.contains("x-request-id"));
    }

    private void testApiDescription(Channel channel, String expected) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/api-description");
        channel.writeAndFlush(req);
        latch.await();

        System.out.println("------------");
        System.out.println(result);
        System.out.println("------------");
        Assert.assertEquals(result, expected);
    }

    private void testDescribeApi(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/noop_v0.1");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, noopApiResult);
    }

    private void testLoadModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop-v0.1&model_name=noop_v0.1&runtime=python");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"noop_v0.1\" registered");
    }

    private void testLoadModelWithInitialWorkers(Channel channel) throws InterruptedException {
        testUnregisterModel(channel);

        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop-v0.1&model_name=noop_v0.1&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workers scaled");
    }

    private void testInitialModelDefaultBatchSettings(Channel channel) throws InterruptedException {
        DescribeModelResponse response = describeModel(channel, "noop-batch");

        Assert.assertEquals(64, response.getBatchSize());
        Assert.assertEquals(300, response.getMaxBatchDelay());
    }

    private void testLoadModelWithBatchSizeOverride(Channel channel) throws InterruptedException {
        unregisterModel(channel, "noop-batch");

        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop-batch-v1.0&batch_size=128&max_batch_delay=250");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"noop-batch\" registered");

        DescribeModelResponse describeResp = describeModel(channel, "noop-batch");

        Assert.assertEquals(128, describeResp.getBatchSize());
        Assert.assertEquals(250, describeResp.getMaxBatchDelay());
    }

    private void testLoadModelWithBatchSize(Channel channel) throws InterruptedException {
        testUnregisterModel(channel);
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop-v0.1&model_name=noop_v0.1&runtime=python&batch_size=256&max_batch_delay=500");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"noop_v0.1\" registered");

        result = null;
        latch = new CountDownLatch(1);
        HttpRequest describeReq =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/noop_v0.1");
        channel.writeAndFlush(describeReq);
        latch.await();

        DescribeModelResponse describeResp = describeModel(channel, "noop_v0.1");

        Assert.assertEquals(256, describeResp.getBatchSize());
        Assert.assertEquals(500, describeResp.getMaxBatchDelay());
    }

    private void testScaleModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/noop_v0.1?min_worker=2");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Processing worker updates...");
    }

    private void testSyncScaleModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/noop_v0.1?synchronous=true&min_worker=1");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workers scaled");
    }

    private void testUnregisterModel(Channel channel) throws InterruptedException {
        StatusResponse resp = unregisterModel(channel, "noop_v0.1");
        Assert.assertEquals(resp.getStatus(), "Model \"noop_v0.1\" unregistered");
    }

    private void testListModels(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models?limit=200&nextPageToken=X");
        channel.writeAndFlush(req);
        latch.await();

        ListModelsResponse resp = JsonUtils.GSON.fromJson(result, ListModelsResponse.class);
        Assert.assertEquals(resp.getModels().size(), 3);
    }

    private void testDescribeModel(Channel channel) throws InterruptedException {
        DescribeModelResponse resp = describeModel(channel, "noop_v0.1");
        Assert.assertTrue(resp.getWorkers().size() > 1);
    }

    private void testPredictions(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        latch.await();
        Assert.assertEquals(result, "OK");
    }

    private void testPredictionsJson(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);

        latch.await();
        Assert.assertEquals(result, "OK");
    }

    private void testPredictionsBinary(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");
        req.content().writeCharSequence("test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testInvocationsJson(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations?model_name=noop");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testInvocationsMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations");

        HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
        encoder.addBodyAttribute("model_name", "noop_v0.1");
        MemoryFileUpload body =
                new MemoryFileUpload("data", "test.txt", "text/plain", null, null, 4);
        body.setContent(Unpooled.copiedBuffer("test", StandardCharsets.UTF_8));
        encoder.addBodyHttpData(body);

        channel.writeAndFlush(encoder.finalizeRequest());
        if (encoder.isChunked()) {
            channel.writeAndFlush(encoder).sync();
        }

        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testPredictionsInvalidRequestSize(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");

        req.content().writeZero(11485760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.REQUEST_ENTITY_TOO_LARGE);
    }

    private void testPredictionsValidRequestSize(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");

        req.content().writeZero(10385760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
    }

    private void testLegacyPredict(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/noop/predict?data=test");
        channel.writeAndFlush(req);

        latch.await();
        Assert.assertEquals(result, "OK");
    }

    private void testInvalidRootRequest() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    private void testInvalidInferenceUri() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    private void testInvalidDescribeModel() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    private void testInvalidPredictionsUri() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    private void testPredictionsModelNotFound() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    private void testInvalidManagementUri() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    private void testInvalidModelsMethod() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    private void testInvalidModelMethod() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    private void testDescribeModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    private void testRegisterModelMissingUrl() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(resp.getMessage(), "Parameter url is required.");
    }

    private void testRegisterModelInvalidRuntime() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=InvalidUrl&runtime=InvalidRuntime");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(resp.getMessage(), "Invalid RuntimeType value: InvalidRuntime");
    }

    private void testRegisterModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models?url=InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found in model store: InvalidUrl");
    }

    private void testRegisterModelMalformedUrl() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=http%3A%2F%2Flocalhost%3Aaaaa");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Invalid model url: http://localhost:aaaa");
    }

    private void testRegisterModelConnectionFailed() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=http%3A%2F%2Flocalhost%3A18888%2Ffake.mar");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download model from: http://localhost:18888/fake.mar");
    }

    private void testRegisterModelHttpError() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=https%3A%2F%2Flocalhost%3A8443%2Ffake.mar");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download model from: https://localhost:8443/fake.mar, code: 404");
    }

    private void testRegisterModelInvalidPath() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models?url=..%2Ffake.mar");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Relative path is not allowed in url: ../fake.mar");
    }

    private void testScaleModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/fake");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: fake");
    }

    private void testUnregisterModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/fake");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: fake");
    }

    private void testScaleModelFailure() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        httpStatus = null;
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=init-error&model_name=init-error");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);

        httpStatus = null;
        result = null;
        latch = new CountDownLatch(1);
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/init-error?synchronous=true&min_worker=1");
        channel.writeAndFlush(req);
        latch.await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(httpStatus, HttpResponseStatus.INTERNAL_SERVER_ERROR);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        Assert.assertEquals(resp.getMessage(), "Failed to start workers");
    }

    private void testInvalidModel() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        httpStatus = null;
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=invalid&model_name=invalid&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse status = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(status.getStatus(), "Workers scaled");

        channel.close();

        channel = connect(false);
        Assert.assertNotNull(channel);

        result = null;
        latch = new CountDownLatch(1);
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/invalid");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        latch.await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(httpStatus, HttpResponseStatus.SERVICE_UNAVAILABLE);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.SERVICE_UNAVAILABLE.code());
        Assert.assertEquals(resp.getMessage(), "Invalid model predict output");
    }

    private void testMetricManager() throws JsonParseException, InterruptedException {
        MetricManager.scheduleMetrics(configManager);
        MetricManager metricManager = MetricManager.getInstance();
        List<Metric> metrics = metricManager.getMetrics();

        // Wait till first value is read in
        int count = 0;
        while (metrics.isEmpty()) {
            Thread.sleep(500);
            metrics = metricManager.getMetrics();
            Assert.assertTrue(++count < 5);
        }
        for (Metric metric : metrics) {
            if (metric.getMetricName().equals("CPUUtilization")) {
                Assert.assertEquals(metric.getUnit(), "Percent");
            }
            if (metric.getMetricName().equals("MemoryUsed")) {
                Assert.assertEquals(metric.getUnit(), "Megabytes");
            }
            if (metric.getMetricName().equals("DiskUsed")) {
                List<Dimension> dimensions = metric.getDimensions();
                for (Dimension dimension : dimensions) {
                    if (dimension.getName().equals("Level")) {
                        Assert.assertEquals(dimension.getValue(), "Host");
                    }
                }
            }
        }
    }

    private DescribeModelResponse describeModel(Channel channel, String modelName)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.GET,
                        String.format("/models/%s", modelName));
        channel.writeAndFlush(req);
        latch.await();

        return JsonUtils.GSON.fromJson(result, DescribeModelResponse.class);
    }

    private StatusResponse unregisterModel(Channel channel, String modelName)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.DELETE,
                        String.format("/models/%s", modelName));
        channel.writeAndFlush(req);
        latch.await();

        return JsonUtils.GSON.fromJson(result, StatusResponse.class);
    }

    private Channel connect(boolean management) {
        Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

        final Connector connector = configManager.getListener(management);
        try {
            Bootstrap b = new Bootstrap();
            final SslContext sslCtx =
                    SslContextBuilder.forClient()
                            .trustManager(InsecureTrustManagerFactory.INSTANCE)
                            .build();
            b.group(Connector.newEventLoopGroup(1))
                    .channel(connector.getClientChannel())
                    .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10000)
                    .handler(
                            new ChannelInitializer<Channel>() {
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    if (connector.isSsl()) {
                                        p.addLast(sslCtx.newHandler(ch.alloc()));
                                    }
                                    p.addLast(new ReadTimeoutHandler(30));
                                    p.addLast(new HttpClientCodec());
                                    p.addLast(new HttpContentDecompressor());
                                    p.addLast(new ChunkedWriteHandler());
                                    p.addLast(new HttpObjectAggregator(6553600));
                                    p.addLast(new TestHandler());
                                }
                            });

            return b.connect(connector.getSocketAddress()).sync().channel();
        } catch (Throwable t) {
            logger.warn("Connect error.", t);
        }
        return null;
    }

    @ChannelHandler.Sharable
    private class TestHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            httpStatus = msg.status();
            result = msg.content().toString(StandardCharsets.UTF_8);
            headers = msg.headers();
            latch.countDown();
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            Logger logger = LoggerFactory.getLogger(TestHandler.class);
            logger.error("Unknown exception", cause);
            ctx.close();
            latch.countDown();
        }
    }
}
