package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.util.ConfigManager;
import io.netty.channel.EventLoopGroup;
import java.util.ArrayList;
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

    public boolean hasNoWorker(String modelName) {
        List<WorkerThread> worker = workers.get(modelName);
        if (worker == null) {
            return true;
        }
        return worker.isEmpty();
    }

    public void modelChanged(Model model) throws WorkerInitializationException {
        int minWorker = model.getMinWorker();
        List<WorkerThread> threads;
        if (minWorker == 0) {
            ModelManager modelManager = ModelManager.getInstance();
            modelManager.getModels().remove(model.getModelName());
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
            BatchAggregator aggregator = new BatchAggregator(configManager, model);
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
