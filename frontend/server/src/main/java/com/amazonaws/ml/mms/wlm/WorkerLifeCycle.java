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

import com.amazonaws.ml.mms.metrics.Metric;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.NettyUtils;
import io.netty.channel.unix.DomainSocketAddress;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerLifeCycle {

    static final Logger logger = LoggerFactory.getLogger(WorkerLifeCycle.class);

    private ConfigManager configManager;
    private Model model;
    private int pid = -1;
    private Process process;
    private CountDownLatch latch;
    private boolean success;
    private int port;

    public WorkerLifeCycle(ConfigManager configManager, Model model) {
        this.configManager = configManager;
        this.model = model;
    }

    private String[] getEnvString(String cwd, String modelPath) {
        ArrayList<String> envList = new ArrayList<>();
        Pattern blackList = configManager.getBlacklistPattern();

        StringBuilder pythonPath = new StringBuilder();
        HashMap<String, String> environment = new HashMap<>(System.getenv());

        if (System.getenv("PYTHONPATH") != null) {
            pythonPath.append(System.getenv("PYTHONPATH")).append(File.pathSeparatorChar);
        }

        pythonPath.append(modelPath).append(File.pathSeparatorChar).append(cwd);

        environment.put("PYTHONPATH", pythonPath.toString());

        for (Map.Entry<String, String> entry : environment.entrySet()) {
            if (!blackList.matcher(entry.getKey()).matches()) {
                envList.add(entry.getKey() + '=' + entry.getValue());
            }
        }

        return envList.toArray(new String[0]); // NOPMD
    }

    public void startWorker(int port) throws WorkerInitializationException, InterruptedException {
        File workingDir = new File(configManager.getModelServerHome());
        File modelPath;
        setPort(port);
        try {
            modelPath = model.getModelDir().getCanonicalFile();
        } catch (IOException e) {
            throw new WorkerInitializationException("Failed get MMS home directory", e);
        }

        SocketAddress address = NettyUtils.getSocketAddress(port);
        String[] args = new String[6];
        args[0] = model.getModelArchive().getManifest().getRuntime().getValue();
        args[1] = new File(workingDir, "mms/model_service_worker.py").getAbsolutePath();
        args[4] = "--sock-type";

        if (address instanceof DomainSocketAddress) {
            args[5] = "unix";
            args[2] = "--sock-name";
            args[3] = ((DomainSocketAddress) address).path();
        } else {
            args[5] = "tcp";
            args[2] = "--port";
            args[3] = String.valueOf(port);
        }

        String[] envp = getEnvString(workingDir.getAbsolutePath(), modelPath.getAbsolutePath());

        try {
            latch = new CountDownLatch(1);

            synchronized (this) {
                process = Runtime.getRuntime().exec(args, envp, modelPath);

                String threadName = "W-" + port;
                new ReaderThread(threadName, process.getErrorStream(), true, this).start();
                new ReaderThread(threadName, process.getInputStream(), false, this).start();
            }

            if (latch.await(2, TimeUnit.MINUTES)) {
                if (!success) {
                    throw new WorkerInitializationException("Backend stream closed.");
                }
                return;
            }
            throw new WorkerInitializationException("Backend worker startup time out.");
        } catch (IOException e) {
            throw new WorkerInitializationException("Failed start worker process", e);
        } finally {
            if (!success) {
                exit();
            }
        }
    }

    public synchronized void exit() {
        if (process != null) {
            process.destroyForcibly();
            process = null;
            SocketAddress address = NettyUtils.getSocketAddress(port);
            if (address instanceof DomainSocketAddress) {
                String path = ((DomainSocketAddress) address).path();
                FileUtils.deleteQuietly(new File(path));
            }
        }
    }

    void setSuccess(boolean success) {
        this.success = success;
        latch.countDown();
    }

    public synchronized int getPid() {
        return pid;
    }

    public synchronized void setPid(int pid) {
        this.pid = pid;
    }

    private synchronized void setPort(int port) {
        this.port = port;
    }

    private static final class ReaderThread extends Thread {

        private InputStream is;
        private boolean error;
        private WorkerLifeCycle lifeCycle;
        static final org.apache.log4j.Logger loggerModelMetrics =
                org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_METRICS_LOGGER);

        public ReaderThread(String name, InputStream is, boolean error, WorkerLifeCycle lifeCycle) {
            super(name + (error ? "-stderr" : "-stdout"));
            this.is = is;
            this.error = error;
            this.lifeCycle = lifeCycle;
        }

        @Override
        public void run() {
            try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
                while (scanner.hasNext()) {
                    String result = scanner.nextLine();
                    if (result == null) {
                        break;
                    }
                    if (result.startsWith("[METRICS]")) {
                        loggerModelMetrics.info(Metric.parse(result.substring(9)));
                        continue;
                    }

                    if ("MxNet worker started.".equals(result)) {
                        lifeCycle.setSuccess(true);
                    } else if (result.startsWith("[PID]")) {
                        lifeCycle.setPid(Integer.parseInt(result.substring("[PID]".length())));
                    }
                    if (error) {
                        logger.warn(result);
                    } else {
                        logger.info(result);
                    }
                }
            } finally {
                lifeCycle.setSuccess(false);
            }
        }
    }
}
