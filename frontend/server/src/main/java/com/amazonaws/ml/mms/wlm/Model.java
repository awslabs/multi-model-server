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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Model {
    public static final String DEFAULT_DATA_QUEUE = "DATA_QUEUE";
    private ModelArchive modelArchive;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    // Total number of subsequent inference request failures
    private AtomicInteger failedInfReqs;
    // Per workerthread job queue. This separates out the control queue from data queue
    private ConcurrentMap<String, LinkedBlockingDeque<Job>> jobsDb;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
        minWorkers = 1;
        maxWorkers = 1;
        batchSize = 1;
        maxBatchDelay = 100;
        jobsDb = new ConcurrentHashMap<>();
        // Always have a queue for data
        jobsDb.putIfAbsent(DEFAULT_DATA_QUEUE, new LinkedBlockingDeque<>(queueSize));
        failedInfReqs = new AtomicInteger(0);
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

    public boolean addJob(String threadId, Job job) {
        jobsDb.putIfAbsent(threadId, new LinkedBlockingDeque<>());
        return jobsDb.get(threadId).offer(job);
    }

    public void removeJobQueue(String threadId) {
        if (!threadId.equals(DEFAULT_DATA_QUEUE)) {
            jobsDb.remove(threadId);
        }
    }

    public boolean addJob(Job job) {
        return addJob(DEFAULT_DATA_QUEUE, job);
    }

    public void addFirst(String threadId, Job job) {
        jobsDb.putIfAbsent(threadId, new LinkedBlockingDeque<>());
        jobsDb.get(threadId).addFirst(job);
    }

    public void addFirst(Job j) {
        addFirst(DEFAULT_DATA_QUEUE, j);
    }

    public Job nextJob(String threadId) throws InterruptedException {
        return nextJob(threadId, Long.MAX_VALUE);
    }

    public Job nextJob(String threadId, long timeout) throws InterruptedException {
        LinkedBlockingDeque<Job> jobs = jobsDb.get(threadId);
        if (jobs != null && !jobs.isEmpty()) {
            Job j = jobs.poll();
            if (j != null) {
                return j;
            }
        }
        return jobsDb.get(DEFAULT_DATA_QUEUE).poll(timeout, TimeUnit.MILLISECONDS);
    }

    public int incrFailedInfReqs() {
        return failedInfReqs.incrementAndGet();
    }

    public void resetFailedInfReqs() {
        failedInfReqs.set(0);
    }
}
