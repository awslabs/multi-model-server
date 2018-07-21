package com.amazonaws.ml.mms.metrics;

import com.amazonaws.ml.mms.TestUtils;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.lang.reflect.Type;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MetricManagerTest {
    static {
        TestUtils.init();
    }

    @Test
    public void test() throws GeneralSecurityException, IOException, JsonParseException {
        ConfigManager configManager = new ConfigManager();
        MetricCollector collector = new MetricCollector(configManager);
        Type listType = new TypeToken<ArrayList<Metric>>() {}.getType();
        Gson gson = new Gson();
        collector.collect();
        String metricJsonString = collector.getJsonString();
        ArrayList<Metric> metrics;
        metrics = gson.fromJson(metricJsonString, listType);
        for (Metric metric : metrics) {
            if (metric.getMetricName().equals("CPUUtilization")) {
                Assert.assertEquals(metric.getUnit(), "Percent");
            }
            if (metric.getMetricName().equals("MemoryUsed")) {
                Assert.assertEquals(metric.getUnit(), "Megabytes");
            }
            if (metric.getMetricName().equals("DiskUsed")) {
                ArrayList<Dimension> dimensions = metric.getDimensions();
                for (Dimension dimension : dimensions) {
                    if (dimension.getName().equals("Level")) {
                        Assert.assertEquals(dimension.getValue(), "Host");
                    }
                }
            }
        }
    }
}
