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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import com.amazonaws.ml.mms.archive.Signature;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Model {

    private ModelArchive modelArchive;
    private int minWorker;
    private int maxWorker;
    private int batchSize;
    private int maxBatchDelay;

    private AtomicInteger numUnsuccessReq;

    private ConcurrentMap<Long, LinkedBlockingDeque<Job>> threadJobs;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
        minWorker = 1;
        maxWorker = 1;
        batchSize = 1;
        maxBatchDelay = 100;
        threadJobs = new ConcurrentHashMap<>();
        threadJobs.put((long) -1, new LinkedBlockingDeque<>());
        numUnsuccessReq = new AtomicInteger(0);
    }

    public final int incrNumUnsuccessReq() {
        return numUnsuccessReq.incrementAndGet();
    }

    public void resetUnsuccessReq() {
        numUnsuccessReq.set(0);
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
        if (threadJobs.get(threadId) == null) {
            threadJobs.put(threadId, new LinkedBlockingDeque<>());
        }
        return threadJobs.get(threadId).offer(job);
    }

    public void removeJobQueue(Long threadId) {
        if (threadId != (long) -1) {
            threadJobs.remove(threadId);
        }
    }

    public boolean addJob(Job job) {
        return addJob(job, (long) -1);
    }

    public void addFirst(Job j, Long threadId) {
        if (threadJobs.get(threadId) == null) {
            threadJobs.put(threadId, new LinkedBlockingDeque<>());
        }
        threadJobs.get(threadId).addFirst(j);
    }

    public Job nextJob(Long threadId) throws InterruptedException {
        Job j;
        if ((threadId != null)
                && (threadJobs.get(threadId) != null)
                && (threadJobs.get(threadId).size() != 0)) {
            j = threadJobs.get(threadId).take();
        } else {
            j = threadJobs.get((long) -1).take();
        }

        return j;
    }

    public Job nextJob(long timeout, Long threadId) throws InterruptedException {
        Job j;
        if ((threadId != null)
                && (threadJobs.get(threadId) != null)
                && (threadJobs.get(threadId).size() != 0)) {
            j = threadJobs.get(threadId).poll(timeout, TimeUnit.MILLISECONDS);
        } else {
            j = threadJobs.get((long) -1).poll(timeout, TimeUnit.MILLISECONDS);
        }
        return j;
    }
}
