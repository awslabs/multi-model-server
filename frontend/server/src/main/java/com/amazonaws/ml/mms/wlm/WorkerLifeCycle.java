package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.util.ConfigManager;
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
    private int gpuId;
    private CountDownLatch latch;
    private boolean success;

    public WorkerLifeCycle(ConfigManager configManager, int gpuId) {
        this.configManager = configManager;
        this.gpuId = gpuId;
    }

    public boolean startWorker(int port, Model model) {
        String[] args = new String[gpuId >= 0 ? 8 : 6];
        args[0] = "python";
        args[1] = "mms/mxnet_worker.py";
        args[2] = "--port";
        args[3] = String.valueOf(port);
        args[4] = "--models";
        args[5] = model.getModelName() + '=' + model.getUrl();
        if (gpuId >= 0) {
            args[6] = "--gpu";
            args[7] = String.valueOf(gpuId);
        }

        File workingDir = new File(configManager.getModelServerHome());
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
                    if ("MxNet worker started.".equals(result)) {
                        lifeCycle.setSuccess(true);
                    }
                    if (error) {
                        logger.error(result);
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
