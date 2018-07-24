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
package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.NettyUtils;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerLifeCycle {

    static final Logger logger = LoggerFactory.getLogger(WorkerLifeCycle.class);
    private ConfigManager configManager;

    private Process process;
    private CountDownLatch latch;
    private boolean success;

    public WorkerLifeCycle(ConfigManager configManager) {
        this.configManager = configManager;
    }

    public boolean startWorker(int port) {
        String[] args = new String[3];
        args[0] = "python";
        args[1] = "mms/model_service_worker.py";
        args[2] = NettyUtils.getSocketAddress(port).toString();

        File workingDir;

        try {
            workingDir = new File(configManager.getModelServerHome()).getCanonicalFile();
        } catch (IOException e) {
            logger.error("Failed start worker process", e);
            return false;
        }

        String pythonPath = System.getenv("PYTHONPATH");
        String pythonEnv;
        if (pythonPath == null || pythonPath.isEmpty()) {
            pythonEnv = "PYTHONPATH=" + workingDir.getAbsolutePath();
        } else {
            pythonEnv =
                    "PYTHONPATH=" + pythonPath + File.pathSeparator + workingDir.getAbsolutePath();
        }
        String[] envp = new String[] {pythonEnv};

        try {
            latch = new CountDownLatch(1);

            synchronized (this) {
                process = Runtime.getRuntime().exec(args, envp, workingDir);

                String threadName = "W-" + port;
                new ReaderThread(threadName, process.getErrorStream(), true, this).start();
                new ReaderThread(threadName, process.getInputStream(), false, this).start();
            }

            if (latch.await(2, TimeUnit.MINUTES)) {
                return success;
            }
            logger.error("Backend worker startup time out.");
            exit();
        } catch (IOException e) {
            logger.error("Failed start worker process", e);
            exit();
        } catch (InterruptedException e) {
            logger.error("Worker process interrupted", e);
            exit();
        }
        return false;
    }

    public synchronized void exit() {
        if (process != null) {
            process.destroyForcibly();
            process = null;
        }
    }

    void setSuccess(boolean success) {
        this.success = success;
        if (!success) {
            exit();
        }
        latch.countDown();
    }

    private static final class ReaderThread extends Thread {

        private InputStream is;
        private boolean error;
        private WorkerLifeCycle lifeCycle;
        static final Logger loggerModelMetrics =
                LoggerFactory.getLogger(ConfigManager.MODEL_METRICS_LOGGER);

        public ReaderThread(String name, InputStream is, boolean error, WorkerLifeCycle lifeCycle) {
            super(name + (error ? "-stderr" : "-stdout"));
            this.is = is;
            this.error = error;
            this.lifeCycle = lifeCycle;
        }

        @Override
        public void run() {
            try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
                boolean metricFound = false;
                StringBuilder jsonString = new StringBuilder();
                while (scanner.hasNext()) {
                    String result = scanner.nextLine();
                    if (result == null) {
                        break;
                    }
                    if ("MxNet worker started.".equals(result)) {
                        lifeCycle.setSuccess(true);
                    }
                    if ("[METRICS]".equals(result) && !metricFound) {
                        metricFound = true;
                        continue;
                    }
                    if (error) {
                        logger.error(result);
                    } else {
                        if (!metricFound) {
                            logger.info(result);
                        } else {
                            if ("[/METRICS]".equals(result)) {
                                loggerModelMetrics.info(jsonString.toString());
                                jsonString.setLength(0);
                                metricFound = false;
                            } else {
                                jsonString.append(result);
                            }
                        }
                    }
                }
            } finally {
                lifeCycle.setSuccess(false);
            }
        }
    }
}
