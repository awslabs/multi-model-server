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
package com.amazonaws.ml.mms.util;

import com.amazonaws.ml.mms.TestUtils;
import com.amazonaws.ml.mms.metrics.Dimension;
import com.amazonaws.ml.mms.metrics.Metric;
import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.List;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

public class ConfigManagerTest {

    static {
        TestUtils.init();
    }

    private Metric createMetric(String metricName, String requestId) {
        List<Dimension> dimensions = new ArrayList<>();
        Metric metric = new Metric();
        metric.setMetricName(metricName);
        metric.setRequestId(requestId);
        metric.setUnit("Milliseconds");
        metric.setTimestamp("1542157988");
        Dimension dimension = new Dimension();
        dimension.setName("Level");
        dimension.setValue("Model");
        dimensions.add(dimension);
        metric.setDimensions(dimensions);
        return metric;
    }

    @Test
    public void test() throws IOException, GeneralSecurityException {
        ConfigManager.Arguments args = new ConfigManager.Arguments();
        args.setModels(new String[] {"noop_v0.1"});
        ConfigManager.init(args);
        ConfigManager configManager = ConfigManager.getInstance();
        configManager.setProperty("keystore", "src/test/resources/keystore.p12");
        Dimension dimension;
        List<Metric> metrics = new ArrayList<>();
        // Create two metrics and add them to a list

        metrics.add(createMetric("TestMetric1", "12345"));
        metrics.add(createMetric("TestMetric2", "23478"));
        org.apache.log4j.Logger logger =
                org.apache.log4j.Logger.getLogger(ConfigManager.MMS_METRICS_LOGGER);
        logger.debug(metrics);
        Assert.assertTrue(new File("build/logs/mms_metrics.log").exists());

        logger = org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_METRICS_LOGGER);
        logger.debug(metrics);
        Assert.assertTrue(new File("build/logs/model_metrics.log").exists());

        Logger modelLogger = LoggerFactory.getLogger(ConfigManager.MODEL_LOGGER);
        modelLogger.debug("test model_log");
        Assert.assertTrue(new File("build/logs/model_log.log").exists());

        SslContext ctx = configManager.getSslContext();
        Assert.assertNotNull(ctx);
    }
}
