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

import com.amazonaws.ml.mms.ModelServer;
import com.amazonaws.ml.mms.util.ConfigManager;
import io.netty.buffer.Unpooled;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.codec.http.multipart.MemoryFileUpload;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.PropertyConfigurator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Cts {

    private byte[] kitten;
    private byte[] player1;
    private byte[] player2;
    private List<String> failedModels;

    private Cts() {
        failedModels = new ArrayList<>();
    }

    public static void main(String[] args) {
        updateLog4jConfiguration();

        Cts cts = new Cts();
        cts.startTest();
    }

    private void startTest() {
        ConfigManager.init(new ConfigManager.Arguments());
        ConfigManager configManager = ConfigManager.getInstance();
        ModelServer server = new ModelServer(configManager);

        Logger logger = LoggerFactory.getLogger(Cts.class);
        try {
            server.start();

            kitten =
                    loadImage(
                            "https://s3.amazonaws.com/model-server/inputs/kitten.jpg",
                            "kitten.jpg");
            player1 =
                    loadImage(
                            "https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/input1.jpg",
                            "player1.jpg");
            player2 =
                    loadImage(
                            "https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/input2.jpg",
                            "player1.jpg");

            HttpClient client = new HttpClient(8081, 8080);

            for (ModelInfo info : ModelInfo.MODEL_ARCHIVE_1) {
                runTest(client, info, logger);
            }

            for (ModelInfo info : ModelInfo.MODEL_ARCHIVE_04) {
                runTest(client, info, logger);
            }
        } catch (Exception e) {
            logger.error("", e);
        } finally {
            try {
                server.stop();
            } catch (Exception e) {
                logger.error("", e);
            }
        }
        if (failedModels.isEmpty()) {
            logger.info("All models passed CTS.");
            System.exit(0);
        } else {
            logger.info("Following models failed CTS:");
            for (String model : failedModels) {
                logger.info(model);
            }
            System.exit(1);
        }
    }

    private void runTest(HttpClient client, ModelInfo info, Logger logger)
            throws HttpPostRequestEncoder.ErrorDataEncoderException, InterruptedException,
                    IOException {
        String modelName = info.getModelName();
        String url = info.getUrl();
        int type = info.getType();

        logger.info("Testing model: {}={}", modelName, url);

        if (!client.registerModel(modelName, url)) {
            failedModels.add(url);
            return;
        }

        try {
            if (!predict(client, type, modelName)) {
                failedModels.add(url);
            }
        } finally {
            if (!client.unregisterModel(modelName)) {
                failedModels.add(url);
            }
        }
    }

    private boolean predict(HttpClient client, int type, String modelName)
            throws HttpPostRequestEncoder.ErrorDataEncoderException, InterruptedException,
                    IOException {
        switch (type) {
            case ModelInfo.FACE_RECOGNITION:
                // arcface
                DefaultFullHttpRequest req =
                        new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/");
                HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
                MemoryFileUpload body =
                        new MemoryFileUpload(
                                "img1", "img1.jpg", "images/jpeg", null, null, player1.length);
                body.setContent(Unpooled.copiedBuffer(player1));
                encoder.addBodyHttpData(body);
                body =
                        new MemoryFileUpload(
                                "img2", "img2.jpg", "images/jpeg", null, null, player2.length);
                body.setContent(Unpooled.copiedBuffer(player2));
                encoder.addBodyHttpData(body);
                return client.predict(modelName, req, encoder);
            case ModelInfo.SEMANTIC_SEGMENTATION:
                // duc
                return client.predict(modelName, kitten, "image/jpeg");
            case ModelInfo.LANGUAGE_MODELING:
                // lstm
                byte[] json =
                        ("[{'input_sentence': 'on the exchange floor as soon"
                                        + " as ual stopped trading we <unk> for a panic"
                                        + " said one top floor trader'}]")
                                .getBytes(StandardCharsets.UTF_8);
                return client.predict(modelName, json, "application/json");
            case ModelInfo.IMAGE_CLASSIFICATION:
            case ModelInfo.EMOTION_DETECTION:
            default:
                return client.predict(modelName, kitten, "image/jpeg");
        }
    }

    private byte[] loadImage(String path, String fileName) throws IOException {
        File file = new File(System.getProperty("java.io.tmpdir"), fileName);
        if (file.exists()) {
            return FileUtils.readFileToByteArray(file);
        }
        byte[] buf = IOUtils.toByteArray(new URL(path));
        FileUtils.writeByteArrayToFile(file, buf);
        return buf;
    }

    private static void updateLog4jConfiguration() {
        System.setProperty("LOG_LOCATION", "logs");
        System.setProperty("METRICS_LOCATION", "logs");

        Properties props = new Properties();
        try (InputStream is = Cts.class.getResourceAsStream("log4j.properties")) {
            props.load(is);
        } catch (IOException e) {
            System.out.println("Cannot load log4j configuration file"); // NOPMD
        }

        PropertyConfigurator.configure(props);
    }
}
