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

import com.amazonaws.ml.mms.archive.InvalidModelException;
import com.amazonaws.ml.mms.archive.Manifest;
import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.common.ErrorCodes;
import com.amazonaws.ml.mms.util.ConfigManager;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager;

    private ConfigManager configManager;
    private WorkLoadManager wlm;
    private ConcurrentHashMap<String, Model> models;

    private ModelManager(ConfigManager configManager, WorkLoadManager wlm) {
        this.configManager = configManager;
        this.wlm = wlm;
        models = new ConcurrentHashMap<>();
    }

    public static void init(ConfigManager configManager, WorkLoadManager wlm) {
        modelManager = new ModelManager(configManager, wlm);
    }

    public static ModelManager getInstance() {
        return modelManager;
    }

    public ModelArchive registerModel(String url) throws InvalidModelException {
        return registerModel(url, null, null, null, 1, 100);
    }

    public ModelArchive registerModel(
            String url,
            String modelName,
            Manifest.RuntimeType runtime,
            String handler,
            int batchSize,
            int maxBatchDelay)
            throws InvalidModelException {
        ModelArchive archive = ModelArchive.downloadModel(configManager.getModelStore(), url);
        if (modelName == null || modelName.isEmpty()) {
            modelName = archive.getModelName();
        } else {
            archive.getManifest().getModel().setModelName(modelName);
        }
        if (runtime != null) {
            archive.getManifest().getEngine().setRuntime(runtime);
        }
        if (handler != null) {
            archive.getManifest().getModel().setHandler(handler);
        }

        Model model = new Model(archive, configManager.getJobQueueSize());
        model.setBatchSize(batchSize);
        model.setMaxBatchDelay(maxBatchDelay);
        Model existingModel = models.putIfAbsent(modelName, model);
        if (existingModel != null) {
            // model already exists
            throw new InvalidModelException(
                    ErrorCodes.MODELS_POST_MODEL_ALREADY_REGISTERED,
                    "Model \"" + modelName + "\" is already registered");
        }
        logger.info("Model {} loaded.", model.getModelName());
        return archive;
    }

    public boolean unregisterModel(String modelName) throws WorkerInitializationException {
        Model model = models.remove(modelName);
        if (model == null) {
            logger.warn("Model not found: " + modelName);
            return false;
        }

        model.setMinWorkers(0);
        model.setMaxWorkers(0);
        wlm.modelChanged(model);
        logger.info("Model {} unregistered.", model.getModelName());
        return true;
    }

    public boolean updateModel(String modelName, int minWorkers, int maxWorkers)
            throws WorkerInitializationException {
        Model model = models.get(modelName);
        if (model == null) {
            logger.warn("Model not found: " + modelName);
            return false;
        }
        model.setMinWorkers(minWorkers);
        model.setMaxWorkers(maxWorkers);
        wlm.modelChanged(model);
        return true;
    }

    public Map<String, Model> getModels() {
        return models;
    }

    public List<WorkerThread> getWorkers(String modelName) {
        return wlm.getWorkers(modelName);
    }

    public HttpResponseStatus addJob(Job job) {
        String modelName = job.getModelName();
        Model model;
        if (modelName == null) {
            if (models.size() != 1) {
                return HttpResponseStatus.NOT_FOUND;
            }
            model = models.entrySet().iterator().next().getValue();
            modelName = model.getModelName();
        } else {
            model = models.get(modelName);
            if (model == null) {
                return HttpResponseStatus.NOT_FOUND;
            }
        }

        if (wlm.hasNoWorker(modelName)) {
            return HttpResponseStatus.SERVICE_UNAVAILABLE;
        }

        if (model.addJob(job)) {
            return HttpResponseStatus.OK;
        }
        return HttpResponseStatus.SERVICE_UNAVAILABLE;
    }
}
