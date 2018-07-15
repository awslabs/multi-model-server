package com.amazonaws.ml.mms.metrics;

import com.amazonaws.ml.mms.TestUtils;

import java.security.GeneralSecurityException;
import org.junit.Assert;

import org.testng.annotations.Test;

public class MetricManagerTest {
    static {
        TestUtils.init();
    }

    @Test
    public void test() throws GeneralSecurityException, InterruptedException {
        MetricManager metricManager = new MetricManager(2000);
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Assert.assertTrue(metricManager.metricJsonString.contains("CPUUtilization"));
        Assert.assertTrue(metricManager.metricStore.map.get("SYSTEM").containsKey("MemoryUsed"));
        Assert.assertTrue(metricManager.metricStore.map.get("SYSTEM").get("MemoryUsed") instanceof Metric);
    }
}
