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
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MetricManager {

    private MetricStore metricStore = new MetricStore();
    private Type metricType = new TypeToken<Map<String, Map<String, Metric>>>() {}.getType();
    private Logger logger = LoggerFactory.getLogger(MetricManager.class);
    private Logger metrics_logger = LoggerFactory.getLogger(ConfigManager.MMS_METRICS_LOGGER);

    public MetricManager(ConfigManager configManager, int timeInterval) {
        Timer t;
        TimerTask timerTask;
        timerTask =
                new TimerTask() {

                    @Override
                    public void run() {
                        // Read and convert
                        MetricCollector collector = new MetricCollector(configManager);
                        Gson gson = new Gson();
                        try {
                            collector.collect();
                            String metricJsonString = collector.getJsonString();
                            metricStore.setMap(gson.fromJson(metricJsonString, metricType));
                            metrics_logger.info(metricJsonString);
                        } catch (IOException | JsonParseException e) {
                            logger.error(e.getMessage());
                        }
                    }
                };
        t = new Timer("Metrics Timer");
        // Every 60 seconds emit system level metrics
        //TODO: Replace time with value in config manager
        t.scheduleAtFixedRate(timerTask, 0, timeInterval);
    }

    public MetricManager(ConfigManager configManager) {
        this(configManager,60000);
    }
}
