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
import io.netty.channel.EventLoopGroup;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkLoadManager {

    private ExecutorService threadPool;

    private ConcurrentHashMap<String, List<WorkerThread>> workers;

    private ConfigManager configManager;
    private EventLoopGroup backendGroup;
    private AtomicInteger port;
    private AtomicInteger gpuCounter;
    private AtomicInteger threadNumber;

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);

    public WorkLoadManager(ConfigManager configManager, EventLoopGroup backendGroup) {
        this.configManager = configManager;
        this.backendGroup = backendGroup;
        this.port = new AtomicInteger(9000);
        this.gpuCounter = new AtomicInteger(0);
        threadPool = Executors.newCachedThreadPool();
        workers = new ConcurrentHashMap<>();
        threadNumber = new AtomicInteger(0);
    }

    public List<WorkerThread> getWorkers(String modelName) {
        List<WorkerThread> list = workers.get(modelName);
        if (list == null) {
            return Collections.emptyList();
        }
        return new ArrayList<>(list);
    }

    public Map<Integer, WorkerThread> getWorkers() {
        Map<Integer, WorkerThread> map = new HashMap<>();
        for (Map.Entry<String, List<WorkerThread>> entry : workers.entrySet()) {
            // Add server thread
            String modelName = entry.getKey();
            List<WorkerThread> workerThreads = entry.getValue();
            WorkerThread serverThread =
                    ModelManager.getInstance().getModels().get(modelName).getServerThread();
            map.put(serverThread.getPid(), serverThread);
            for (WorkerThread worker : workerThreads) {
                map.put(worker.getPid(), worker);
            }
        }
        return map;
    }

    public boolean hasNoWorker(String modelName) {
        List<WorkerThread> worker = workers.get(modelName);
        if (worker == null) {
            return true;
        }
        return worker.isEmpty();
    }

    public int getNumRunningWorkers(String modelName) {
        int numWorking = 0;
        List<WorkerThread> threads = workers.getOrDefault(modelName, null);

        if (threads != null) {
            for (WorkerThread thread : threads) {
                if ((thread.getState() != WorkerState.WORKER_STOPPED)
                        && (thread.getState() != WorkerState.WORKER_ERROR)
                        && (thread.getState() != WorkerState.WORKER_SCALED_DOWN)) {
                    numWorking += 1;
                }
            }
        }

        return numWorking;
    }

    public CompletableFuture<HttpResponseStatus> modelChanged(Model model) {
        synchronized (model.getModelName()) {
            CompletableFuture<HttpResponseStatus> future = new CompletableFuture<>();
            int minWorker = model.getMinWorkers();
            int maxWorker = model.getMaxWorkers();
            List<WorkerThread> threads;
            if (minWorker == 0) {
                threads = workers.remove(model.getModelName());
                if (threads == null) {
                    if (maxWorker == 0) {
                        return shutdownServerThread(model, future);
                    }
                    future.complete(HttpResponseStatus.OK);
                    return future;
                }
            } else {
                threads = workers.computeIfAbsent(model.getModelName(), k -> new ArrayList<>());
            }

            int currentWorkers = threads.size();
            if (currentWorkers < minWorker) {
                addThreads(threads, model, minWorker - currentWorkers, future);
            } else {
                for (int i = currentWorkers - 1; i >= maxWorker; --i) {
                    WorkerThread thread = threads.remove(i);
                    thread.shutdown();
                }
                if (maxWorker == 0) {
                    return shutdownServerThread(model, future);
                }
                future.complete(HttpResponseStatus.OK);
            }
            return future;
        }
    }

    private CompletableFuture<HttpResponseStatus> shutdownServerThread(
            Model model, CompletableFuture<HttpResponseStatus> future) {
        model.getServerThread().shutdown();
        WorkerLifeCycle lifecycle = model.getServerThread().getLifeCycle();
        Process workerProcess = lifecycle.getProcess();
        if (workerProcess.isAlive()) {
            boolean workerDestroyed = false;
            workerProcess.destroy(); // sends SIGTERM to workerProcess
            try {
                workerDestroyed =
                        workerProcess.waitFor(
                                configManager.getUnregisterModelTimeout(), TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                logger.warn(
                        "WorkerThread interrupted during waitFor, possible asynch resource cleanup.");
                future.complete(HttpResponseStatus.INTERNAL_SERVER_ERROR);
                return future;
            }
            if (!workerDestroyed) {
                logger.warn("WorkerThread timed out while cleaning, please resend request.");
                future.complete(HttpResponseStatus.REQUEST_TIMEOUT);
                return future;
            }
        }
        future.complete(HttpResponseStatus.OK);
        return future;
    }

    public void addServerThread(Model model, CompletableFuture<HttpResponseStatus> future)
            throws InterruptedException, ExecutionException, TimeoutException {
        WorkerStateListener listener = new WorkerStateListener(future, 1);
        BatchAggregator aggregator = new BatchAggregator(model);
        synchronized (model.getModelName()) {
            model.setPort(port.getAndIncrement());
            WorkerThread thread =
                    new WorkerThread(
                            configManager,
                            backendGroup,
                            model.getPort(),
                            -1,
                            model,
                            aggregator,
                            listener,
                            threadNumber.getAndIncrement(),
                            true);
            model.setServerThread(thread);
            threadPool.submit(thread);
            future.get(1, TimeUnit.MINUTES);
        }
    }

    private void addThreads(
            List<WorkerThread> threads,
            Model model,
            int count,
            CompletableFuture<HttpResponseStatus> future) {
        WorkerStateListener listener = new WorkerStateListener(future, count);
        int maxGpu = configManager.getNumberOfGpu();
        for (int i = 0; i < count; ++i) {
            int gpuId = -1;

            if (maxGpu > 0) {
                gpuId = gpuCounter.accumulateAndGet(maxGpu, (prev, maxGpuId) -> ++prev % maxGpuId);
            }

            BatchAggregator aggregator = new BatchAggregator(model);
            WorkerThread thread =
                    new WorkerThread(
                            configManager,
                            backendGroup,
                            model.getPort(),
                            gpuId,
                            model,
                            aggregator,
                            listener,
                            threadNumber.getAndIncrement(),
                            false);

            threads.add(thread);
            threadPool.submit(thread);
        }
    }

    public void scheduleAsync(Runnable r) {
        threadPool.execute(r);
    }
}
