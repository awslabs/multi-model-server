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
import java.io.File;
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

    // Per worker thread job queue. This separates out the control queue from data queue
    private ConcurrentMap<String, LinkedBlockingDeque<Job>> jobsDb;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
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

    public File getModelDir() {
        return modelArchive.getModelDir();
    }

    public String getModelUrl() {
        return modelArchive.getUrl();
    }

    public ModelArchive getModelArchive() {
        return modelArchive;
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

    public void addJob(String threadId, Job job) {
        LinkedBlockingDeque<Job> blockingDeque = jobsDb.get(threadId);
        if (blockingDeque == null) {
            blockingDeque = new LinkedBlockingDeque<>();
            jobsDb.put(threadId, blockingDeque);
        }
        blockingDeque.offer(job);
    }

    public void removeJobQueue(String threadId) {
        if (!threadId.equals(DEFAULT_DATA_QUEUE)) {
            jobsDb.remove(threadId);
        }
    }

    public boolean addJob(Job job) {
        return jobsDb.get(DEFAULT_DATA_QUEUE).offer(job);
    }

    public void addFirst(Job job) {
        jobsDb.get(DEFAULT_DATA_QUEUE).addFirst(job);
    }

    public Job nextJob(String threadId) throws InterruptedException {
        return nextJob(threadId, Long.MAX_VALUE);
    }

    public Job nextJob(String threadId, long timeoutMillis) throws InterruptedException {
        LinkedBlockingDeque<Job> jobs = jobsDb.get(threadId);
        if (jobs != null && !jobs.isEmpty()) {
            Job j = jobs.poll();
            if (j != null) {
                return j;
            }
        }
        return jobsDb.get(DEFAULT_DATA_QUEUE).poll(timeoutMillis, TimeUnit.MILLISECONDS);
    }

    public int incrFailedInfReqs() {
        return failedInfReqs.incrementAndGet();
    }

    public void resetFailedInfReqs() {
        failedInfReqs.set(0);
    }
}
