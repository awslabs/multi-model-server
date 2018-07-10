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
import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.util.ConfigManager;
import io.netty.handler.codec.http.HttpResponseStatus;
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
        ModelArchive archive = ModelArchive.downloadModel(configManager.getModelStore(), url);
        String modelName = archive.getModelName();
        Model model = new Model(archive, configManager.getJobQueueSize());
        Model existingModel = models.putIfAbsent(modelName, model);
        if (existingModel != null) {
            // model already exists
            throw new InvalidModelException("Model have been registered already: " + modelName);
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

        model.setMinWorker(0);
        model.setMaxWorker(0);
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
        model.setMinWorker(minWorkers);
        model.setMaxWorker(maxWorkers);
        wlm.modelChanged(model);
        return true;
    }

    public Map<String, Model> getModels() {
        return models;
    }

    public HttpResponseStatus addJob(Job job) {
        Payload payload = job.getPayload();
        String modelName = payload.getId();
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
