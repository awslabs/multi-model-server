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
package com.amazonaws.ml.mms.metrics;

import com.amazonaws.ml.mms.util.ConfigManager;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public final class MetricManager {

    private static MetricManager metricManager;
    private List<Metric> metrics;

    private MetricManager() {}

    public static synchronized MetricManager getInstance() {
        if (metricManager == null) {
            metricManager = new MetricManager();
        }
        return metricManager;
    }

    public static void scheduleMetrics(ConfigManager configManager) {
        MetricCollector metricCollector = new MetricCollector(configManager);
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        scheduler.scheduleAtFixedRate(
                metricCollector, 0, configManager.getMetricTimeInterval(), TimeUnit.SECONDS);
    }

    public synchronized List<Metric> getMetrics() {
        return metrics;
    }

    public synchronized void setMetrics(List<Metric> metrics) {
        this.metrics = metrics;
    }
}
