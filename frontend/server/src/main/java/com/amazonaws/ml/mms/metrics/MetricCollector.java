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

    private static final Logger logger = LoggerFactory.getLogger(MetricCollector.class);
    private ConfigManager configManager;
    private static final Logger loggerMetrics =
            LoggerFactory.getLogger(ConfigManager.MMS_METRICS_LOGGER);

    public MetricCollector(ConfigManager configManager) {
        this.configManager = configManager;
    }

    public String collect() throws IOException {
        String jsonString;
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
                    "PYTHONPATH=" + pythonPath + File.pathSeparator + workingDir.getAbsolutePath();
        }
        // sbin added for macs for python sysctl pythonpath
        String path = System.getenv("PATH");
        String osName = System.getProperty("os.name");
        if (osName.startsWith("Mac OS X")) {
            StringBuilder pathBuilder = new StringBuilder();
            pathBuilder.append("PATH=");
            pathBuilder.append(path);
            pathBuilder.append(File.pathSeparator);
            pathBuilder.append("/sbin/");
            path = pathBuilder.toString();
        }
        String[] env = new String[] {pythonEnv, path};
        Process p = Runtime.getRuntime().exec(args, env, workingDir);
        InputStream stdOut = p.getInputStream();

        InputStream stdErr = p.getErrorStream();
        Scanner scanner = new Scanner(stdOut, StandardCharsets.UTF_8.name());

        // read the output from the command
        while (scanner.hasNext()) {
            stringBuilder.append(scanner.nextLine());
        }
        jsonString = stringBuilder.toString();
        // read any errors from the attempted command

        scanner = new Scanner(stdErr, StandardCharsets.UTF_8.name());
        if (scanner.hasNext()) {
            StringBuilder error = new StringBuilder();
            error.append("Error while running system metrics script:\n");
            while (scanner.hasNext()) {
                error.append(scanner.nextLine());
            }
            throw new IOException(error.toString());
        }
        return jsonString;
    }

    @Override
    public void run() {
        Gson gson = new Gson();
        try {
            String metricJsonString = collect();
            Type listType = new TypeToken<ArrayList<Metric>>() {}.getType();
            MetricManager metricManager = MetricManager.getInstance();
            metricManager.setMetrics(gson.fromJson(metricJsonString, listType));
            loggerMetrics.info(metricJsonString);
        } catch (Exception e) {
            logger.error(e.getMessage());
        }
    }
}
