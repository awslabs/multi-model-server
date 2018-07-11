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

import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.archive.Signature;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

public class Model {

    private ModelArchive modelArchive;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private LinkedBlockingDeque<Job> jobs;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
        minWorkers = 1;
        maxWorkers = 1;
        batchSize = 1;
        maxBatchDelay = 100;
        jobs = new LinkedBlockingDeque<>(queueSize);
    }

    public String getModelName() {
        return modelArchive.getModelName();
    }

    public String getModelDir() {
        return modelArchive.getModelDir().getAbsolutePath();
    }

    public String getModelUrl() {
        return modelArchive.getUrl();
    }

    public ModelArchive getModelArchive() {
        return modelArchive;
    }

    public String getRequestContentType() {
        Signature signature = modelArchive.getSignature();
        if (signature == null) {
            return null;
        }
        Map<String, List<Signature.Parameter>> request = signature.getRequest();
        if (request.isEmpty()) {
            return null;
        }
        return request.keySet().iterator().next();
    }

    public String getResponseContentType() {
        Signature signature = modelArchive.getSignature();
        if (signature == null) {
            return null;
        }
        Map<String, List<Signature.Parameter>> resp = signature.getResponse();
        if (resp.isEmpty()) {
            return null;
        }
        return resp.keySet().iterator().next();
    }

    public int getMinWorkers() {
        return minWorkers;
    }

    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorkers;
    }

    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    public boolean addJob(Job job) {
        return jobs.offer(job);
    }

    public void addFirst(Job j) {
        jobs.addFirst(j);
    }

    public Job nextJob() throws InterruptedException {
        return jobs.take();
    }

    public Job nextJob(long timeout) throws InterruptedException {
        return jobs.poll(timeout, TimeUnit.MILLISECONDS);
    }
}
