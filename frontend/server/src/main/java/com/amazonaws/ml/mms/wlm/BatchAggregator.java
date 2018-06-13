package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.util.ConfigManager;
import io.netty.handler.codec.http.HttpHeaderValues;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(BatchAggregator.class);

    private ConfigManager configManager;
    private Model model;
    private Map<String, Job> jobs;

    public BatchAggregator(ConfigManager configManager, Model model) {
        this.configManager = configManager;
        this.model = model;
        jobs = new HashMap<>();
    }

    public Message getRequest() throws InterruptedException {
        jobs.clear();

        // first job is a blocking call;
        Job job = model.nextJob();
        jobs.put(job.getJobId(), job);

        logger.debug("get first job: {}", job.getJobId());

        long maxBatchDelay = configManager.getMaxBatchDelay();
        int size = configManager.getMaxBatchSize() - 1;
        long begin = System.currentTimeMillis();
        for (int i = 0; i < size; ++i) {

            job = model.nextJob(maxBatchDelay);
            if (job == null) {
                break;
            }
            jobs.put(job.getJobId(), job);
            long end = System.currentTimeMillis();
            maxBatchDelay -= end - begin;
            begin = end;
            if (maxBatchDelay <= 0) {
                break;
            }
        }

        logger.debug("sending jobs, size: {}", jobs.size());

        Message message = new Message(model.getModelName());
        for (Job j : jobs.values()) {
            Payload payload = j.getPayload();
            Payload p = new Payload(j.getJobId(), payload.getData());
            message.addPayload(p);
        }

        return message;
    }

    public void sendResponse(Message message) {
        logger.debug("received response, size: {}", message.getPayloads().size());

        for (Payload payload : message.getPayloads()) {
            String jobId = payload.getId();
            Job job = jobs.remove(jobId);
            if (job == null) {
                throw new IllegalStateException("Unexpected job: " + jobId);
            }
            job.response(payload.getData(), model.getResponseContentType());
        }
        if (!jobs.isEmpty()) {
            throw new IllegalStateException("Not all jobs get response.");
        }
    }

    public void sendError(Message message, String error) {
        byte[] body = error.getBytes(StandardCharsets.UTF_8);
        for (Payload payload : message.getPayloads()) {
            String jobId = payload.getId();
            Job job = jobs.remove(jobId);
            if (job == null) {
                throw new IllegalStateException("Unexpected job: " + jobId);
            }
            job.response(body, HttpHeaderValues.APPLICATION_JSON);
        }
        if (!jobs.isEmpty()) {
            throw new IllegalStateException("Not all jobs get response.");
        }
    }
}
