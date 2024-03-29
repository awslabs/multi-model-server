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

import com.amazonaws.ml.mms.archive.Manifest;
import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.archive.ModelException;
import com.amazonaws.ml.mms.archive.ModelNotFoundException;
import com.amazonaws.ml.mms.http.ConflictStatusException;
import com.amazonaws.ml.mms.http.StatusResponse;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.NettyUtils;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeoutException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager;

    private ConfigManager configManager;
    private WorkLoadManager wlm;
    private ConcurrentHashMap<String, Model> models;
    private HashSet<String> startupModels;
    private ScheduledExecutorService scheduler;

    private ModelManager(ConfigManager configManager, WorkLoadManager wlm) {
        this.configManager = configManager;
        this.wlm = wlm;
        models = new ConcurrentHashMap<>();
        scheduler = Executors.newScheduledThreadPool(2);
        this.startupModels = new HashSet<>();
    }

    public ScheduledExecutorService getScheduler() {
        return scheduler;
    }

    public static void init(ConfigManager configManager, WorkLoadManager wlm) {
        modelManager = new ModelManager(configManager, wlm);
    }

    public static ModelManager getInstance() {
        return modelManager;
    }

    public ModelArchive registerModel(String url, String defaultModelName, String preloadModel)
            throws ModelException, IOException, InterruptedException, ExecutionException,
                    TimeoutException {
        return registerModel(
                url,
                null,
                null,
                null,
                1,
                100,
                configManager.getDefaultResponseTimeoutSeconds(),
                defaultModelName,
                preloadModel);
    }

    public ModelArchive registerModel(
            String url,
            String modelName,
            Manifest.RuntimeType runtime,
            String handler,
            int batchSize,
            int maxBatchDelay,
            int responseTimeoutSeconds,
            String defaultModelName,
            String preloadModel)
            throws ModelException, IOException, InterruptedException, ExecutionException,
                    TimeoutException {

        ModelArchive archive = ModelArchive.downloadModel(configManager.getModelStore(), url);
        if (modelName == null || modelName.isEmpty()) {
            if (archive.getModelName() == null || archive.getModelName().isEmpty()) {
                archive.getManifest().getModel().setModelName(defaultModelName);
            }
            modelName = archive.getModelName();
        } else {
            archive.getManifest().getModel().setModelName(modelName);
        }
        if (runtime != null) {
            archive.getManifest().setRuntime(runtime);
        }
        if (handler != null) {
            archive.getManifest().getModel().setHandler(handler);
        } else if (archive.getHandler() == null || archive.getHandler().isEmpty()) {
            archive.getManifest()
                    .getModel()
                    .setHandler(configManager.getMmsDefaultServiceHandler());
        }

        archive.validate();

        Model model = new Model(archive, configManager.getJobQueueSize(), preloadModel);
        model.setBatchSize(batchSize);
        model.setMaxBatchDelay(maxBatchDelay);
        model.setResponseTimeoutSeconds(responseTimeoutSeconds);
        Model existingModel = models.putIfAbsent(modelName, model);
        if (existingModel != null) {
            // model already exists
            throw new ConflictStatusException("Model " + modelName + " is already registered.");
        }

        if (configManager.isDebug()) {
            model.setPort(9000);
        } else {
            startBackendServer(model);
        }

        models.put(modelName, model);

        logger.info("Model {} loaded.", model.getModelName());

        return archive;
    }

    public HttpResponseStatus unregisterModel(String modelName) {
        Model model = models.remove(modelName);
        if (model == null) {
            logger.warn("Model not found: " + modelName);
            return HttpResponseStatus.NOT_FOUND;
        }
        model.setMinWorkers(0);
        model.setMaxWorkers(0);
        CompletableFuture<HttpResponseStatus> futureStatus = wlm.modelChanged(model);
        HttpResponseStatus httpResponseStatus = HttpResponseStatus.OK;

        try {
            httpResponseStatus = futureStatus.get();
        } catch (InterruptedException | ExecutionException e) {
            logger.warn("Process was interrupted while cleaning resources.");
            httpResponseStatus = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        // Only continue cleaning if resource cleaning succeeded
        if (httpResponseStatus == HttpResponseStatus.OK) {
            model.getModelArchive().clean();
            startupModels.remove(modelName);
            logger.info("Model {} unregistered.", modelName);
        } else {
            models.put(modelName, model);
        }

        return httpResponseStatus;
    }

    public void startBackendServer(Model model)
            throws InterruptedException, ExecutionException, TimeoutException {
        CompletableFuture<HttpResponseStatus> future = new CompletableFuture<>();
        if (model == null) {
            throw new AssertionError("Model not found");
        }
        wlm.addServerThread(model, future);
    }

    public CompletableFuture<HttpResponseStatus> updateModel(
            String modelName, int minWorkers, int maxWorkers) {
        Model model = models.get(modelName);
        if (model == null) {
            throw new AssertionError("Model not found: " + modelName);
        }
        model.setMinWorkers(minWorkers);
        model.setMaxWorkers(maxWorkers);
        logger.debug("updateModel: {}, count: {}", modelName, minWorkers);
        return wlm.modelChanged(model);
    }

    public Map<String, Model> getModels() {
        return models;
    }

    public List<WorkerThread> getWorkers(String modelName) {
        return wlm.getWorkers(modelName);
    }

    public Map<Integer, WorkerThread> getWorkers() {
        return wlm.getWorkers();
    }

    public boolean addJob(Job job) throws ModelNotFoundException {
        String modelName = job.getModelName();
        Model model = models.get(modelName);
        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }

        if (wlm.hasNoWorker(modelName)) {
            return false;
        }

        return model.addJob(job);
    }

    public void workerStatus(final ChannelHandlerContext ctx) {
        Runnable r =
                () -> {
                    String response = "Healthy";
                    int numWorking = 0;
                    int numScaled = 0;
                    for (Map.Entry<String, Model> m : models.entrySet()) {
                        numScaled += m.getValue().getMinWorkers();
                        numWorking += wlm.getNumRunningWorkers(m.getValue().getModelName());
                    }

                    if ((numWorking > 0) && (numWorking < numScaled)) {
                        response = "Partial Healthy";
                    } else if ((numWorking == 0) && (numScaled > 0)) {
                        response = "Unhealthy";
                    }

                    // TODO: Check if its OK to send other 2xx errors to ALB for "Partial Healthy"
                    // and "Unhealthy"
                    NettyUtils.sendJsonResponse(
                            ctx, new StatusResponse(response), HttpResponseStatus.OK);
                };
        wlm.scheduleAsync(r);
    }

    public boolean scaleRequestStatus(String modelName) {
        Model model = ModelManager.getInstance().getModels().get(modelName);
        int numWorkers = wlm.getNumRunningWorkers(modelName);

        return model == null || model.getMinWorkers() <= numWorkers;
    }

    public void submitTask(Runnable runnable) {
        wlm.scheduleAsync(runnable);
    }

    public Set<String> getStartupModels() {
        return startupModels;
    }
}
