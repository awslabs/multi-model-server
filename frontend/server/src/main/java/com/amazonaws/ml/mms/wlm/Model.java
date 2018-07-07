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
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

public class Model {

    private ModelArchive modelArchive;
    private int minWorker;
    private int maxWorker;
    private LinkedBlockingDeque<Job> jobs;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
        minWorker = 1;
        maxWorker = 1;
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
        return modelArchive.getSignature().getRequest().getContentType();
    }

    public String getResponseContentType() {
        return modelArchive.getSignature().getResponse().getContentType();
    }

    public int getMinWorker() {
        return minWorker;
    }

    public void setMinWorker(int minWorker) {
        this.minWorker = minWorker;
    }

    public int getMaxWorker() {
        return maxWorker;
    }

    public void setMaxWorker(int maxWorker) {
        this.maxWorker = maxWorker;
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
