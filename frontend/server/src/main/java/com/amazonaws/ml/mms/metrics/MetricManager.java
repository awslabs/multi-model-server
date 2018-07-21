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
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.ArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MetricManager implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(MetricManager.class);
    private static final Logger loggerMetrics =
            LoggerFactory.getLogger(ConfigManager.MMS_METRICS_LOGGER);
    private MetricCollector collector;
    private ArrayList<Metric> metrics;

    @Override
    public void run() {
        Gson gson = new Gson();
        try {
            String metricJsonString = collector.collect();
            Type listType = new TypeToken<ArrayList<Metric>>() {}.getType();
            setMetrics(gson.fromJson(metricJsonString, listType));
            loggerMetrics.info(metricJsonString);
        } catch (Exception e) {
            logger.error(e.getMessage());
        }
    }

    public MetricManager(ConfigManager configManager) {
        collector = new MetricCollector(configManager);
    }

    public ArrayList<Metric> getMetrics() {
        return metrics;
    }

    public void setMetrics(ArrayList<Metric> metrics) {
        this.metrics = metrics;
    }
}
