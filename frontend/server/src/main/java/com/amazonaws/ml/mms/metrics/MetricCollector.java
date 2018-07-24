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
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Scanner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MetricCollector implements Runnable {

    static final Logger logger = LoggerFactory.getLogger(MetricCollector.class);
    private static final Logger loggerMetrics =
            LoggerFactory.getLogger(ConfigManager.MMS_METRICS_LOGGER);
    private static Type listType = new TypeToken<ArrayList<Metric>>() {}.getType();

    private ConfigManager configManager;

    public MetricCollector(ConfigManager configManager) {
        this.configManager = configManager;
    }

    public String collect() throws IOException, InterruptedException {
        StringBuilder stringBuilder = new StringBuilder();
        String[] args = new String[2];
        args[0] = "python";
        args[1] = "mms/metrics/system_metrics.py";
        // run the Unix "python script to collect metrics" command
        // using the Runtime exec method:

        File workingDir;

        try {
            workingDir = new File(configManager.getModelServerHome()).getCanonicalFile();
        } catch (IOException e) {
            logger.error("Failed to run system metrics script", e);
            return "{}";
        }

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
        String[] env = new String[] {pythonEnv, path.toString()};
        Process p = Runtime.getRuntime().exec(args, env, workingDir);
        InputStream stdOut = p.getInputStream();

        InputStream stdErr = p.getErrorStream();
        Scanner scanner = new Scanner(stdOut, StandardCharsets.UTF_8.name());
        while (scanner.hasNext()) {
            stringBuilder.append(scanner.nextLine());
        }
        new ReaderThread("MetricCollectorError", stdErr).start();

        return stringBuilder.toString();
    }

    @Override
    public void run() {
        Gson gson = new Gson();
        try {
            String metricJsonString = collect();
            MetricManager metricManager = MetricManager.getInstance();
            metricManager.setMetrics(gson.fromJson(metricJsonString, listType));
            loggerMetrics.info(metricJsonString);
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
        }
    }

    private static final class ReaderThread extends Thread {

        private InputStream is;

        public ReaderThread(String name, InputStream is) {
            super(name + "-stderr");
            this.is = is;
        }

        @Override
        public void run() {
            Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name());
            StringBuilder stringBuilder = new StringBuilder();
            while (scanner.hasNext()) {
                String result = scanner.nextLine();
                if (result == null) {
                    break;
                }
                stringBuilder.append(result);
            }
            if (!stringBuilder.toString().isEmpty()) {
                logger.error(stringBuilder.toString());
            }
        }
    }
}
