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
import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.wlm.ModelManager;
import com.amazonaws.ml.mms.wlm.WorkerThread;
import com.google.gson.reflect.TypeToken;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MetricCollector implements Runnable {

    static final Logger logger = LoggerFactory.getLogger(MetricCollector.class);
    private static final org.apache.log4j.Logger loggerMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MMS_METRICS_LOGGER);
    private static final Type LIST_TYPE = new TypeToken<ArrayList<Metric>>() {}.getType();
    private ConfigManager configManager;

    public MetricCollector(ConfigManager configManager) {
        this.configManager = configManager;
    }

    @Override
    public void run() {
        try {
            // Collect System level Metrics
            String[] args = new String[2];
            args[0] = "python";
            args[1] = "mms/metrics/metric_collector.py";
            File workingDir = new File(configManager.getModelServerHome()).getCanonicalFile();

            String pythonPath = System.getenv("PYTHONPATH");
            String pythonEnv;
            if (pythonPath == null || pythonPath.isEmpty()) {
                pythonEnv = "PYTHONPATH=" + workingDir.getAbsolutePath();
            } else {
                pythonEnv =
                        "PYTHONPATH="
                                + pythonPath
                                + File.pathSeparatorChar
                                + workingDir.getAbsolutePath();
            }
            // sbin added for macs for python sysctl pythonpath
            StringBuilder path = new StringBuilder();
            path.append("PATH=").append(System.getenv("PATH"));
            String osName = System.getProperty("os.name");
            if (osName.startsWith("Mac OS X")) {
                path.append(File.pathSeparatorChar).append("/sbin/");
            }
            String[] env = {pythonEnv, path.toString()};
            final Process p = Runtime.getRuntime().exec(args, env, workingDir);

            ModelManager modelManager = ModelManager.getInstance();
            Map<Integer, WorkerThread> workerMap = modelManager.getWorkers();
            try (OutputStream os = p.getOutputStream()) {
                writeWorkerPids(workerMap, os);
            }

            new Thread(
                            () -> {
                                try {
                                    String error =
                                            IOUtils.toString(
                                                    p.getErrorStream(), StandardCharsets.UTF_8);
                                    if (!error.isEmpty()) {
                                        logger.error(error);
                                    }
                                } catch (IOException e) {
                                    logger.error("", e);
                                }
                            })
                    .start();

            MetricManager metricManager = MetricManager.getInstance();
            try (BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                // first line is system metrics
                String line = reader.readLine();
                if (line == null || line.isEmpty()) {
                    logger.error("Expecting system metrics line, but received empty.");
                    return;
                }

                List<Metric> metricsSystem = JsonUtils.GSON.fromJson(line, LIST_TYPE);
                metricManager.setMetrics(metricsSystem);
                loggerMetrics.info(metricsSystem);

                // Collect process level metrics
                while ((line = reader.readLine()) != null) {
                    String[] tokens = line.split(":");
                    if (tokens.length != 2) {
                        continue;
                    }

                    Integer pid = Integer.valueOf(tokens[0]);
                    WorkerThread worker = workerMap.get(pid);
                    worker.setMemory(Long.parseLong(tokens[1]));
                }
            }
        } catch (IOException e) {
            logger.error("", e);
        }
    }

    private void writeWorkerPids(Map<Integer, WorkerThread> workerMap, OutputStream os)
            throws IOException {
        boolean first = true;
        for (Integer pid : workerMap.keySet()) {
            if (pid < 0) {
                logger.warn("worker pid is not available yet.");
                continue;
            }
            if (first) {
                first = false;
            } else {
                IOUtils.write(",", os, StandardCharsets.UTF_8);
            }
            IOUtils.write(pid.toString(), os, StandardCharsets.UTF_8);
        }
    }
}
