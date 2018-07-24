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
import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.io.IOException;
import java.security.GeneralSecurityException;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

public class ConfigManagerTest {

    static {
        TestUtils.init();
    }

    @Test
    public void test() throws IOException, GeneralSecurityException {
        ConfigManager configManager = new ConfigManager();
        configManager.setProperty("keystore", "src/test/resources/keystore.p12");

        Logger logger = LoggerFactory.getLogger(ConfigManager.MMS_METRICS_LOGGER);
        logger.debug("test mms_metrics");
        Assert.assertTrue(new File("build/logs/mms_metrics.log").exists());

        logger = LoggerFactory.getLogger(ConfigManager.MODEL_METRICS_LOGGER);
        logger.debug("test model_metrics");
        Assert.assertTrue(new File("build/logs/model_metrics.log").exists());

        logger = LoggerFactory.getLogger(ConfigManager.MODEL_LOGGER);
        logger.debug("test model_log");
        Assert.assertTrue(new File("build/logs/model_log.log").exists());

        SslContext ctx = configManager.getSslContext();
        Assert.assertNotNull(ctx);
    }
}
