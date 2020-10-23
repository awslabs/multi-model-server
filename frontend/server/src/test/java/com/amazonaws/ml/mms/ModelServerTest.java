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
import com.amazonaws.ml.mms.protobuf.codegen.InferenceRequest;
import com.amazonaws.ml.mms.protobuf.codegen.InputParameter;
import com.amazonaws.ml.mms.protobuf.codegen.RequestInput;
import com.amazonaws.ml.mms.protobuf.codegen.WorkerCommands;
import com.amazonaws.ml.mms.servingsdk.impl.PluginsManager;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.Connector;
import com.amazonaws.ml.mms.util.JsonUtils;
import com.google.gson.JsonParseException;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
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
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.ByteArrayOutputStream;
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
    ByteBuffer resultBuf;
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
        PluginsManager.getInstance().initialize();

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
                    IOException, NoSuchFieldException, IllegalAccessException {
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
        testPingProto(channel);

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
        testLoadModelWithInitialWorkers(managementChannel);
        testLoadModelWithInitialWorkersWithJSONReqBody(managementChannel);
        testPredictions(channel);
        testPredictionsBinary(channel);
        testPredictionsJson(channel);
        testPredictionsProto();
        testInvocationsJson(channel);
        testInvocationsMultipart(channel);
        testModelsInvokeJson(channel);
        testModelsInvokeMultipart(channel);
        testLegacyPredict(channel);
        testPredictionsInvalidRequestSize(channel);
        testPredictionsValidRequestSize(channel);
        testPredictionsDecodeRequest(channel, managementChannel);
        testPredictionsDoNotDecodeRequest(channel, managementChannel);
        testPredictionsModifyResponseHeader(channel, managementChannel);
        testPredictionsNoManifest(channel, managementChannel);
        testModelRegisterWithDefaultWorkers(managementChannel);
        testLogging(channel, managementChannel);
        testLoggingUnload(channel, managementChannel);
        testLoadingMemoryError();
        testPredictionMemoryError();
        testMetricManager();
        testErrorBatch();

        channel.close();
        managementChannel.close();

        // negative test case, channel will be closed by server
        testInvalidRootRequest();
        testInvalidInferenceUri();
        testInvalidPredictionsUri();
        testInvalidDescribeModel();
        testPredictionsModelNotFound();
        testPredictionsModelNotFoundProto();
        testInvalidManagementUri();
        testInvalidModelsMethod();
        testInvalidModelMethod();
        testDescribeModelNotFound();
        testRegisterModelMissingUrl();
        testRegisterModelInvalidRuntime();
        testRegisterModelNotFound();
        testRegisterModelConflict();
        testRegisterModelMalformedUrl();
        testRegisterModelConnectionFailed();
        testRegisterModelHttpError();
        testRegisterModelInvalidPath();
        testScaleModelNotFound();
        testScaleModelFailure();
        testUnregisterModelNotFound();
        testUnregisterModelTimeout();
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

    private void testPingProto(Channel channel)
            throws InterruptedException, InvalidProtocolBufferException {
        resultBuf = null;
        latch = new CountDownLatch(1);
        InferenceRequest inferenceRequest =
                InferenceRequest.newBuilder().setCommand(WorkerCommands.ping).build();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/ping");
        req.headers().add("Content-Type", ConfigManager.HTTP_CONTENT_TYPE_PROTOBUF);
        req.content().writeBytes(inferenceRequest.toByteArray());
        channel.writeAndFlush(req);
        latch.await();

        com.amazonaws.ml.mms.protobuf.codegen.StatusResponse resp =
                com.amazonaws.ml.mms.protobuf.codegen.StatusResponse.parseFrom(resultBuf);
        Assert.assertEquals(resp.getMessage(), "Healthy");
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
                        "/models?url=noop-v0.1&model_name=noop_v0.1&runtime=python&synchronous=false");
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

    private void testLoadModelWithInitialWorkersWithJSONReqBody(Channel channel)
            throws InterruptedException {
        testUnregisterModel(channel);

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        req.headers().add("Content-Type", "application/json");
        req.content()
                .writeCharSequence(
                        "{'url':'noop-v0.1', 'model_name':'noop_v0.1', 'initial_workers':'1', 'synchronous':'true'}",
                        CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workers scaled");
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
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/noop_v0.1");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
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
        Assert.assertEquals(resp.getModels().size(), 2);
    }

    private void testDescribeModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/noop_v0.1");
        channel.writeAndFlush(req);
        latch.await();

        DescribeModelResponse resp = JsonUtils.GSON.fromJson(result, DescribeModelResponse.class);
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

    private void testPredictionsProto() throws InterruptedException, IOException {
        Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

        float[] featureVec = {
            0.8241127f, 0.77719664f, 0.47123995f, 0.27323001f, 0.24874457f, 0.77869387f,
            0.50711921f, 0.10696663f, 0.60663805f, 0.76063525f, 0.96358908f, 0.71026102f,
            0.57714464f, 0.58250422f, 0.91595038f, 0.24119576f, 0.58981158f, 0.67119473f,
            0.94832165f, 0.91711728f, 0.0323646f, 0.07007003f, 0.89158581f, 0.01916486f,
            0.5647568f, 0.99879008f, 0.58311515f, 0.87001143f, 0.50620349f, 0.65268692f,
            0.83657373f, 0.31589474f, 0.70910797f, 0.62886395f, 0.03498501f, 0.36503007f,
            0.94178899f, 0.21739391f, 0.29688258f, 0.34630696f, 0.30494259f, 0.04302086f,
            0.3578226f, 0.04361075f, 0.91962488f, 0.24961093f, 0.0124245f, 0.31004002f,
            0.61543447f, 0.34500444f, 0.30441186f, 0.44085924f, 0.67489625f, 0.03938287f,
            0.89307169f, 0.22283647f, 0.44441515f, 0.82044036f, 0.37541783f, 0.25868981f,
            0.46510721f, 0.51640271f, 0.40917042f, 0.65912921f, 0.72228879f, 0.42611241f,
            0.71283259f, 0.37417586f, 0.786403f, 0.6912011f, 0.4338622f, 0.29868897f,
            0.0342538f, 0.16938266f, 0.90234809f, 0.3051922f, 0.92377579f, 0.97883088f,
            0.2028601f, 0.50478822f, 0.84762944f, 0.11011502f, 0.70006246f, 0.34329564f,
            0.49022718f, 0.8569296f, 0.75698334f, 0.84864789f, 0.9477985f, 0.46994381f,
            0.05319027f, 0.07369953f, 0.08497094f, 0.54536333f, 0.87922514f, 0.97857665f,
            0.06930542f, 0.27101086f, 0.03069235f, 0.13432096f, 0.96021588f, 0.9484153f,
            0.75365465f, 0.76216408f, 0.43294879f, 0.41034781f, 0.01088872f, 0.29060839f,
            0.94462721f, 0.83999491f, 0.4364634f, 0.63611379f, 0.32102346f, 0.10418961f,
            0.2776194f, 0.73166493f, 0.76387601f, 0.83429646f, 0.94348065f, 0.85956626f,
            0.81160069f, 0.1650624f, 0.79505978f, 0.67288331f, 0.3204887f, 0.89388283f,
            0.85290859f, 0.11308228f, 0.81252801f, 0.87276483f, 0.76737167f, 0.16166891f,
            0.78767838f, 0.79160494f, 0.80843258f, 0.39723985f, 0.47062281f, 0.96028728f,
            0.55309858f, 0.05378428f, 0.3619188f, 0.69888766f, 0.76134346f, 0.60911425f,
            0.85562674f, 0.58098788f, 0.5438003f, 0.61229528f, 0.14350196f, 0.75286178f,
            0.88131248f, 0.69132185f, 0.12576858f, 0.23459534f, 0.26883056f, 0.98129534f,
            0.74060036f, 0.9607236f, 0.99617814f, 0.75829678f, 0.06310486f, 0.55572225f,
            0.72709395f, 0.77374732f, 0.81625695f, 0.13475297f, 0.89352917f, 0.19805313f,
            0.34789188f, 0.08422005f, 0.67733949f, 0.94300965f, 0.22116594f, 0.10948816f,
            0.50651639f, 0.40402931f, 0.46181863f, 0.14743327f, 0.33300708f, 0.87358395f,
            0.79312213f, 0.54662338f, 0.83890467f, 0.87690315f, 0.24570711f, 0.01534696f,
            0.11803501f, 0.21333099f, 0.75169896f, 0.42758898f, 0.80780874f, 0.57331851f,
            0.96341639f, 0.52078203f, 0.22610806f, 0.83348684f, 0.76036637f, 0.99407179f,
            0.96098997f, 0.2451298f, 0.41848766f, 0.01584927f, 0.28213452f, 0.04494721f,
            0.16963578f, 0.68096619f, 0.39404686f, 0.7621266f, 0.02721071f, 0.5481559f,
            0.59972178f, 0.61725009f, 0.76405802f, 0.83030081f, 0.87232659f, 0.16119207f,
            0.51143718f, 0.13040968f, 0.57453206f, 0.63200166f, 0.27077547f, 0.72281371f,
            0.44055048f, 0.51538986f, 0.29096202f, 0.99726975f, 0.50958807f, 0.87792484f,
            0.03956957f, 0.42187308f, 0.87694541f, 0.88974026f, 0.65590356f, 0.35029236f,
            0.18853136f, 0.50500502f, 0.95545852f, 0.94636341f, 0.84731837f, 0.13936297f,
            0.32537976f, 0.41430316f, 0.18574781f, 0.97574309f, 0.26483325f, 0.79840404f,
            0.74069621f, 0.98526361f, 0.63957011f, 0.30924823f, 0.20429374f, 0.09850504f,
            0.77676228f, 0.40561045f, 0.71999222f, 0.42545573f, 0.78092917f, 0.74532941f,
            0.52263514f, 0.01771433f, 0.15041333f, 0.41157879f, 0.15047035f, 0.66149007f,
            0.95970903f, 0.97348663f, 0.30155038f, 0.06596597f, 0.3317747f, 0.09346482f,
            0.71672818f, 0.13279156f, 0.19758743f, 0.20143709f, 0.84517665f, 0.767672f,
            0.21471986f, 0.75663108f, 0.35878468f, 0.58943601f, 0.98005496f, 0.30451585f,
            0.34754926f, 0.3298018f, 0.36859658f, 0.52568727f, 0.45107675f, 0.27778918f,
            0.4825746f, 0.6521011f, 0.16924284f, 0.54550222f, 0.33862934f, 0.88247624f,
            0.97012639f, 0.64496125f, 0.09514454f, 0.90497989f, 0.82705286f, 0.5232794f,
            0.80558394f, 0.86949601f, 0.78825486f, 0.23086437f, 0.64405503f, 0.02989425f,
            0.61423185f, 0.45341492f, 0.52462891f, 0.93029992f, 0.74040612f, 0.45227326f,
            0.35339424f, 0.30661544f, 0.70083487f, 0.68725394f, 0.2036894f, 0.85478822f,
            0.13176267f, 0.10494695f, 0.17226407f, 0.88662847f, 0.42744141f, 0.44540842f,
            0.94161152f, 0.46699513f, 0.36795051f, 0.0234292f, 0.68830582f, 0.33571055f,
            0.93930267f, 0.76513689f, 0.69002036f, 0.11983312f, 0.05524331f, 0.28743821f,
            0.53563344f, 0.00152629f, 0.50295284f, 0.24351331f, 0.6770774f, 0.42484211f,
            0.10956752f, 0.01239354f, 0.57630947f, 0.16575461f, 0.7870273f, 0.64387019f,
            0.65514058f, 0.62808722f, 0.29263556f, 0.8159863f, 0.18642033f
        };
        List<float[]> instances = new LinkedList<>();
        Random rand = new Random();
        for (int i = 0; i < 50; i++) {
            float[] data = new float[featureVec.length];
            for (int j = 0; j < featureVec.length; j++) {
                data[j] = featureVec[rand.nextInt(featureVec.length)];
            }
            instances.add(data);
        }

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeObject(instances);
        byte[] bytes = bos.toByteArray();
        byte[] byteString = ByteString.copyFrom(bytes).toByteArray();

        InputParameter parameter =
                InputParameter.newBuilder().setValue(ByteString.copyFrom(bytes)).build();
        InferenceRequest inferenceRequest =
                InferenceRequest.newBuilder()
                        .setCommand(WorkerCommands.predictions)
                        .setModelName("test")
                        .setRequest(RequestInput.newBuilder().addParameters(parameter).build())
                        .build();
        logger.info(
                "2D random float size="
                        + featureVec.length * 50
                        + ", byteString size="
                        + byteString.length
                        + ", bytes size="
                        + bytes.length
                        + ", parameter size="
                        + parameter.toByteArray().length
                        + ", protobuf size="
                        + inferenceRequest.toByteArray().length);
        oos.close();

        List<float[]> instances1 = new LinkedList<>();
        for (int i = 0; i < 50; i++) {
            instances1.add(featureVec);
        }

        ByteArrayOutputStream bos1 = new ByteArrayOutputStream();
        ObjectOutputStream oos1 = new ObjectOutputStream(bos1);
        oos1.writeObject(instances1);
        byte[] bytes1 = bos1.toByteArray();
        byte[] byteString1 = ByteString.copyFrom(bytes1).toByteArray();

        InputParameter parameter1 =
                InputParameter.newBuilder().setValue(ByteString.copyFrom(bytes1)).build();
        InferenceRequest inferenceRequest1 =
                InferenceRequest.newBuilder()
                        .setCommand(WorkerCommands.predictions)
                        .setModelName("test")
                        .setRequest(RequestInput.newBuilder().addParameters(parameter1).build())
                        .build();
        logger.info(
                "2D repeated float size="
                        + featureVec.length * 50
                        + ", byteString size="
                        + byteString1.length
                        + ", bytes size="
                        + bytes1.length
                        + ", parameter size="
                        + parameter1.toByteArray().length
                        + ", protobuf size="
                        + inferenceRequest1.toByteArray().length);
        oos1.close();

        ByteBuffer fBuffer = ByteBuffer.allocate(Float.BYTES * featureVec.length * 50);
        fBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 50; i++) {
            for (float feature : featureVec) {
                fBuffer.putFloat(feature);
            }
        }
        byte[] bytes2 = fBuffer.array();
        InputParameter parameter2 =
                InputParameter.newBuilder().setValue(ByteString.copyFrom(bytes2)).build();
        InferenceRequest inferenceRequest2 =
                InferenceRequest.newBuilder()
                        .setCommand(WorkerCommands.predictions)
                        .setModelName("test")
                        .setRequest(RequestInput.newBuilder().addParameters(parameter2).build())
                        .build();
        logger.info(
                "1D repeated float size="
                        + featureVec.length * 50
                        + ", fBuffer size="
                        + fBuffer.array().length
                        + ", bytes size="
                        + bytes2.length
                        + ", parameter size="
                        + parameter2.toByteArray().length
                        + ", protobuf size="
                        + inferenceRequest2.toByteArray().length);
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

    private void testModelsInvokeJson(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop/invoke");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testModelsInvokeMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop/invoke");

        HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
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

    private void loadTests(Channel channel, String model, String modelName)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String url =
                "/models?url="
                        + model
                        + "&model_name="
                        + modelName
                        + "&initial_workers=1&synchronous=true";
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        channel.writeAndFlush(req);
        latch.await();
    }

    private void unloadTests(Channel channel, String modelName) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String expected = "Model \"" + modelName + "\" unregistered";
        String url = "/models/" + modelName;
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, url);
        channel.writeAndFlush(req);
        latch.await();
        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), expected);
    }

    private void setConfiguration(String key, String val)
            throws NoSuchFieldException, IllegalAccessException {
        Field f = configManager.getClass().getDeclaredField("prop");
        f.setAccessible(true);
        Properties p = (Properties) f.get(configManager);
        p.setProperty(key, val);
    }

    private void testModelRegisterWithDefaultWorkers(Channel mgmtChannel)
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
        setConfiguration("default_workers_per_model", "1");
        loadTests(mgmtChannel, "noop-v1.0", "noop_default_model_workers");

        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/noop_default_model_workers");
        mgmtChannel.writeAndFlush(req);

        latch.await();
        DescribeModelResponse resp = JsonUtils.GSON.fromJson(result, DescribeModelResponse.class);
        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertEquals(resp.getMinWorkers(), 1);
        unloadTests(mgmtChannel, "noop_default_model_workers");
        setConfiguration("default_workers_per_model", "0");
    }

    private void testPredictionsDecodeRequest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        setConfiguration("decode_input_request", "true");
        loadTests(mgmtChannel, "noop-v1.0-config-tests", "noop-config");

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop-config");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertFalse(result.contains("bytearray"));
        unloadTests(mgmtChannel, "noop-config");
    }

    private void testPredictionsDoNotDecodeRequest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        setConfiguration("decode_input_request", "false");
        loadTests(mgmtChannel, "noop-v1.0-config-tests", "noop-config");

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop-config");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertTrue(result.contains("bytearray"));
        unloadTests(mgmtChannel, "noop-config");
    }

    private void testPredictionsModifyResponseHeader(
            Channel inferChannel, Channel managementChannel)
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
        setConfiguration("decode_input_request", "false");
        loadTests(managementChannel, "respheader-test", "respheader");
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/respheader");

        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertEquals(headers.get("dummy"), "1");
        Assert.assertEquals(headers.get("content-type"), "text/plain");
        Assert.assertTrue(result.contains("bytearray"));
        unloadTests(managementChannel, "respheader");
    }

    private void testPredictionsNoManifest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        setConfiguration("default_service_handler", "service:handle");
        loadTests(mgmtChannel, "noop-no-manifest", "nomanifest");

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/nomanifest");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertEquals(result, "OK");
        unloadTests(mgmtChannel, "nomanifest");
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

    private void testPredictionsModelNotFoundProto()
            throws InterruptedException, InvalidProtocolBufferException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        resultBuf = null;
        latch = new CountDownLatch(1);

        InputParameter parameter =
                InputParameter.newBuilder()
                        .setName("data")
                        .setValue(ByteString.copyFrom("test", CharsetUtil.UTF_8))
                        .build();
        InferenceRequest inferenceRequest =
                InferenceRequest.newBuilder()
                        .setCommand(WorkerCommands.predictions)
                        .setModelName("InvalidModel")
                        .setRequest(RequestInput.newBuilder().addParameters(parameter).build())
                        .build();

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/");
        req.content().writeBytes(inferenceRequest.toByteArray());
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, ConfigManager.HTTP_CONTENT_TYPE_PROTOBUF);
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        latch.await();

        com.amazonaws.ml.mms.protobuf.codegen.StatusResponse resp =
                com.amazonaws.ml.mms.protobuf.codegen.StatusResponse.parseFrom(resultBuf);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
        channel.close();
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

    private void testRegisterModelConflict() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop-v0.1&model_name=noop_v0.1&runtime=python&synchronous=false");
        channel.writeAndFlush(req);
        latch.await();

        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop-v0.1&model_name=noop_v0.1&runtime=python&synchronous=false");
        channel.writeAndFlush(req);
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.CONFLICT.code());
        Assert.assertEquals(resp.getMessage(), "Model noop_v0.1 is already registered.");
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
                        "/models?url=http%3A%2F%2Flocalhost%3A18888%2Ffake.mar&synchronous=false");
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
                        "/models?url=https%3A%2F%2Flocalhost%3A8443%2Ffake.mar&synchronous=false");
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
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=..%2Ffake.mar&synchronous=false");
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

    private void testUnregisterModelTimeout()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel channel = connect(true);
        setConfiguration("unregister_model_timeout", "0");

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/noop_v0.1");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.REQUEST_TIMEOUT.code());
        Assert.assertEquals(resp.getMessage(), "Timed out while cleaning resources: noop_v0.1");

        channel = connect(true);
        setConfiguration("unregister_model_timeout", "120");

        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/noop_v0.1");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
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
                        "/models?url=init-error&model_name=init-error&synchronous=false");
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

    private void testLoadingMemoryError() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=loading-memory-error&model_name=memory_error&runtime=python&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close();
    }

    private void testPredictionMemoryError() throws InterruptedException {
        // Load the model
        Channel channel = connect(true);
        Assert.assertNotNull(channel);
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=prediction-memory-error&model_name=pred-err&runtime=python&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();
        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        channel.close();

        // Test for prediction
        channel = connect(false);
        Assert.assertNotNull(channel);
        result = null;
        latch = new CountDownLatch(1);
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/pred-err");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);

        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close();

        // Unload the model
        channel = connect(true);
        httpStatus = null;
        latch = new CountDownLatch(1);
        Assert.assertNotNull(channel);
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/pred-err");
        channel.writeAndFlush(req);
        latch.await();
        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
    }

    private void testErrorBatch() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        httpStatus = null;
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=error_batch&model_name=err_batch&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse status = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(status.getStatus(), "Workers scaled");

        channel.close();

        channel = connect(false);
        Assert.assertNotNull(channel);

        result = null;
        latch = new CountDownLatch(1);
        httpStatus = null;
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/err_batch");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.INSUFFICIENT_STORAGE);
        Assert.assertEquals(result, "Invalid response");
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

    private void testLogging(Channel inferChannel, Channel mgmtChannel)
            throws NoSuchFieldException, IllegalAccessException, InterruptedException, IOException {
        setConfiguration("default_workers_per_model", "2");
        loadTests(mgmtChannel, "logging", "logging");
        int niter = 5;
        int expected = 2;
        for (int i = 0; i < niter; i++) {
            latch = new CountDownLatch(1);
            DefaultFullHttpRequest req =
                    new DefaultFullHttpRequest(
                            HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/logging");
            req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
            HttpUtil.setContentLength(req, req.content().readableBytes());
            req.headers()
                    .set(
                            HttpHeaderNames.CONTENT_TYPE,
                            HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
            inferChannel.writeAndFlush(req);
            latch.await();
            Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        }

        File logfile = new File("build/logs/mms_log.log");
        Assert.assertTrue(logfile.exists());
        Scanner logscanner = new Scanner(logfile, "UTF-8");
        int count = 0;
        while (logscanner.hasNextLine()) {
            String line = logscanner.nextLine();
            if (line.contains("LoggingService inference [PID]:")) {
                count = count + 1;
            }
        }
        Logger logger = LoggerFactory.getLogger(ModelServerTest.class);
        logger.info("testLogging, found {}, min expected {}.", count, expected);
        Assert.assertTrue(count >= expected);
        unloadTests(mgmtChannel, "logging");
    }

    private void testLoggingUnload(Channel inferChannel, Channel mgmtChannel)
            throws NoSuchFieldException, IllegalAccessException, InterruptedException, IOException {
        setConfiguration("default_workers_per_model", "2");
        loadTests(mgmtChannel, "logging", "logging");
        unloadTests(mgmtChannel, "logging");
        int expected = 1;
        int count = 0;
        File logfile = new File("build/logs/mms_log.log");
        Assert.assertTrue(logfile.exists());
        Scanner logscanner = new Scanner(logfile, "UTF-8");
        while (logscanner.hasNextLine()) {
            String line = logscanner.nextLine();
            if (line.contains("Model logging unregistered")) {
                count = count + 1;
            }
        }
        Assert.assertTrue(count >= expected);
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
            CharSequence contentType = HttpUtil.getMimeType(msg);
            if (contentType != null
                    && contentType
                            .toString()
                            .equalsIgnoreCase(ConfigManager.HTTP_CONTENT_TYPE_PROTOBUF)) {
                resultBuf = msg.content().nioBuffer();
            } else {
                result = msg.content().toString(StandardCharsets.UTF_8);
            }
            httpStatus = msg.status();
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
