package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.archive.ModelArchive;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

public class Model {

    private ModelArchive modelArchive;
    private int minWorker;
    private int maxWorker;
    private ArrayBlockingQueue<Job> jobs;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
        minWorker = 1;
        maxWorker = 1;
        jobs = new ArrayBlockingQueue<>(queueSize);
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

    public Job nextJob() throws InterruptedException {
        return jobs.take();
    }

    public Job nextJob(long timeout) throws InterruptedException {
        return jobs.poll(timeout, TimeUnit.MILLISECONDS);
    }
}
