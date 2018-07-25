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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class WorkLoadManager {

    private ExecutorService threadPool;
    private ConcurrentHashMap<String, List<WorkerThread>> workers;

    private ConfigManager configManager;
    private EventLoopGroup backendGroup;
    private int port = 9000;
    private int gpuCounter;

    public WorkLoadManager(ConfigManager configManager, EventLoopGroup backendGroup) {
        this.configManager = configManager;
        this.backendGroup = backendGroup;
        threadPool = Executors.newCachedThreadPool();
        workers = new ConcurrentHashMap<>();
    }

    public List<WorkerThread> getWorkers(String modelName) {
        List<WorkerThread> list = workers.get(modelName);
        if (list == null) {
            return Collections.emptyList();
        }
        return new ArrayList<>(list);
    }

    public boolean hasNoWorker(String modelName) {
        List<WorkerThread> worker = workers.get(modelName);
        if (worker == null) {
            return true;
        }
        return worker.isEmpty();
    }

    public void modelChanged(Model model) throws WorkerInitializationException {
        int minWorker = model.getMinWorkers();
        List<WorkerThread> threads;
        if (minWorker == 0) {
            threads = workers.remove(model.getModelName());
            if (threads == null) {
                return;
            }
        } else {
            threads = workers.computeIfAbsent(model.getModelName(), k -> new ArrayList<>());
        }

        int currentWorkers = threads.size();
        if (currentWorkers < minWorker) {
            addThreads(threads, model, minWorker - currentWorkers);
        } else {
            for (int i = currentWorkers - 1; i >= minWorker; --i) {
                WorkerThread thread = threads.remove(i);
                thread.shutdown();
            }
        }
    }

    private void addThreads(List<WorkerThread> threads, Model model, int count)
            throws WorkerInitializationException {
        int maxGpu = configManager.getNumberOfGpu();
        for (int i = 0; i < count; ++i) {
            int gpuId = -1;
            if (maxGpu > 0) {
                gpuId = gpuCounter;
                if (++gpuCounter >= maxGpu) {
                    gpuCounter = 0;
                }
            }
            BatchAggregator aggregator = new BatchAggregator(model);
            WorkerThread thread =
                    new WorkerThread(
                            configManager, threads, backendGroup, port, gpuId, model, aggregator);
            thread.connect();
            threads.add(thread);
            threadPool.submit(thread);
            if (!configManager.isDebug()) {
                port++;
            }
        }
    }
}
