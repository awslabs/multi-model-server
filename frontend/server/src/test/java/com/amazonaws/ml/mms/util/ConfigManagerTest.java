package com.amazonaws.ml.mms.util;

import io.netty.handler.ssl.SslContext;
import java.io.IOException;
import java.security.GeneralSecurityException;
import org.junit.Assert;
import org.testng.annotations.Test;

public class ConfigManagerTest {

    @Test
    public void test() throws IOException, GeneralSecurityException {
        ConfigManager configManager = new ConfigManager();
        configManager.setProperty("keystore", "src/test/resources/keystore.p12");

        SslContext ctx = configManager.getSslContext();
        Assert.assertNotNull(ctx);
    }
}
