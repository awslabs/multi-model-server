package com.amazonaws.ml.mms.metrics;

import com.amazonaws.ml.mms.TestUtils;
import com.amazonaws.ml.mms.util.ConfigManager;

import java.io.IOException;
import java.lang.reflect.Type;
import java.security.GeneralSecurityException;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MetricManagerTest {
    static {
        TestUtils.init();
    }

    @Test
    public void test() throws GeneralSecurityException, IOException, JsonParseException {
        ConfigManager configManager = new ConfigManager();
        Type metricType = new TypeToken<Map<String, Map<String, Metric>>>() {}.getType();
        MetricCollector collector = new MetricCollector(configManager);
        MetricStore metricStore = new MetricStore();
        Gson gson = new Gson();
        collector.collect();
        String metricJsonString = collector.getJsonString();
        metricStore.setMap(gson.fromJson(metricJsonString, metricType));
        Map localMap = metricStore.getMap();
        Assert.assertTrue(localMap.containsKey("SYSTEM"));
        Map metricsMap = (Map) localMap.get("SYSTEM");
        Assert.assertTrue(metricsMap.containsKey("CPUUtilization"));
        Assert.assertTrue(metricsMap.containsKey("MemoryUsed"));
        Assert.assertTrue(metricsMap.containsKey("DiskUsage"));
        Metric testMetric = (Metric) metricsMap.get("MemoryUsed");
        Assert.assertTrue(testMetric.getUnit().equals("Megabytes"));

    }
}
