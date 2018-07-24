package com.amazonaws.ml.mms.metrics;

import com.amazonaws.ml.mms.TestUtils;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.google.gson.JsonParseException;
import java.security.GeneralSecurityException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MetricManagerTest {
    static {
        TestUtils.init();
    }

    @Test
    public void test() throws GeneralSecurityException, JsonParseException, InterruptedException {
        ConfigManager configManager = new ConfigManager();
        MetricManager.scheduleMetrics(configManager);
        MetricManager metricManager = MetricManager.getInstance();
        List<Metric> metrics;
        metrics = metricManager.getMetrics();
        // Wait till first value is read in
        while (metrics == null) {
            Thread.sleep(500);
            metrics = metricManager.getMetrics();
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
}
