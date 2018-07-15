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

    private ModelArchive modelArchive;
    private int minWorker;
    private int maxWorker;
    private int batchSize;
    private int maxBatchDelay;

    // numFailedInfReqs is to used to determine number of unsuccessful starts for backend worker.
    // Currently success is measured as any response coming back from the backend-worker. This can be changed
    // to a inference request coming from backend worker in the future.
    private AtomicInteger numFailedInfReqs;

    private ConcurrentMap<Long, LinkedBlockingDeque<Job>> jobsDb;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
        minWorker = 1;
        maxWorker = 1;
        batchSize = 1;
        maxBatchDelay = 100;
        jobsDb = new ConcurrentHashMap<>();
        // Set a queue for data
        jobsDb.put(WorkerThread.DEFAULT_THREAD_ID, new LinkedBlockingDeque<>());
        numFailedInfReqs = new AtomicInteger(0);
    }

    public final int incrNumFailedInfReq() {
        return numFailedInfReqs.incrementAndGet();
    }

    public void resetNumFailedInfReqs() {
        numFailedInfReqs.set(0);
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
        return minWorker;
    }

    public void setMinWorkers(int minWorkers) {
        this.minWorker = minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorker;
    }

    public void setMaxWorkers(int maxWorkers) {
        this.maxWorker = maxWorkers;
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

    public boolean addJob(Job job, Long threadId) {
        if (jobsDb.get(threadId) == null) {
            jobsDb.put(threadId, new LinkedBlockingDeque<>());
        }
        return jobsDb.get(threadId).offer(job);
    }

    public void removeJobQueue(Long threadId) {
        if (threadId != WorkerThread.DEFAULT_THREAD_ID) {
            jobsDb.remove(threadId);
        }
    }

    public boolean addJob(Job job) {
        return addJob(job, WorkerThread.DEFAULT_THREAD_ID);
    }

    public void addFirst(Job j, Long threadId) {
        if (jobsDb.get(threadId) == null) {
            jobsDb.put(threadId, new LinkedBlockingDeque<>());
        }
        jobsDb.get(threadId).addFirst(j);
    }

    public Job nextJob(Long threadId) throws InterruptedException {
        Job j;
        if ((threadId != null)
                && (jobsDb.get(threadId) != null)
                && (jobsDb.get(threadId).size() != 0)) {
            j = jobsDb.get(threadId).take();
        } else {
            j = jobsDb.get(WorkerThread.DEFAULT_THREAD_ID).take();
        }

        return j;
    }

    public Job nextJob(long timeout, Long threadId) throws InterruptedException {
        Job j;
        if ((threadId != null)
                && (jobsDb.get(threadId) != null)
                && (jobsDb.get(threadId).size() != 0)) {
            j = jobsDb.get(threadId).poll(timeout, TimeUnit.MILLISECONDS);
        } else {
            j = jobsDb.get(WorkerThread.DEFAULT_THREAD_ID).poll(timeout, TimeUnit.MILLISECONDS);
        }
        return j;
    }
}
