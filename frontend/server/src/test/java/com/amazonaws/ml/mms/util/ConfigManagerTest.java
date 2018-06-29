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

        logger = LoggerFactory.getLogger(ConfigManager.MMS_METRICS_LOGGER);
        logger.debug("test model_metrics");
        Assert.assertTrue(new File("build/logs/model_metrics.log").exists());

        logger = LoggerFactory.getLogger(ConfigManager.MODEL_LOGGER);
        logger.debug("test model_log");
        Assert.assertTrue(new File("build/logs/model_log.log").exists());

        SslContext ctx = configManager.getSslContext();
        Assert.assertNotNull(ctx);
    }
}
