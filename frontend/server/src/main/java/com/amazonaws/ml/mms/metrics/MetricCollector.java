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
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MetricCollector {

    static final Logger logger = LoggerFactory.getLogger(MetricCollector.class);
    private String jsonString;
    private ConfigManager configManager;

    public MetricCollector(ConfigManager configManager) {
        this.configManager = configManager;
    }

    public void collect() throws IOException {

        StringBuilder stringBuilder = new StringBuilder();
        String[] args = new String[2];
        args[0] = "python";
        args[1] = "mms/system_metrics.py";
        // run the Unix "python script to collect metrics" command
        // using the Runtime exec method:

        File workingDir;

        try {
            workingDir = new File(configManager.getModelServerHome()).getCanonicalFile();
        } catch (IOException e) {
            logger.error("Failed to run system metrics script", e);
            return;
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
        path = "PATH=" + path + File.pathSeparator + "/sbin/";
        String[] env = new String[] {pythonEnv, path};
        Process p = Runtime.getRuntime().exec(args, env, workingDir);
        InputStream stdOut = p.getInputStream();

        InputStream stdErr = p.getErrorStream();
        Scanner scanner = new Scanner(stdOut, StandardCharsets.UTF_8.name());

        // read the output from the command
        while (scanner.hasNext()) {
            stringBuilder.append(scanner.nextLine());
        }
        setJsonString(stringBuilder.toString());
        // read any errors from the attempted command

        scanner = new Scanner(stdErr, StandardCharsets.UTF_8.name());
        if (scanner.hasNext()) {
            throw new IOException("Error while running system metrics script");
        }
    }

    public String getJsonString() {
        return jsonString;
    }

    public void setJsonString(String jsonString) {
        this.jsonString = jsonString;
    }
}
