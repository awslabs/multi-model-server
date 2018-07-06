package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.messages.AbstractRequest;
import com.amazonaws.ml.mms.util.messages.ModelInferenceRequest;
import com.amazonaws.ml.mms.util.messages.ModelInputs;
import com.amazonaws.ml.mms.util.messages.ModelLoadRequest;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.Predictions;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.ListIterator;
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
        jobs = new LinkedHashMap<>();
    }

    public AbstractRequest getRequest() throws InterruptedException {
        jobs.clear();

        // first job is a blocking call;
        Job job = model.nextJob();
        if (job.isControlCmd()) {
            ModelLoadRequest req = new ModelLoadRequest(model.getModelName());
            req.setModelPath(model.getModelUrl());
            return req;
        }

        jobs.put(job.getJobId(), job);

        logger.debug("get first job: {}", job.getJobId());

        long maxBatchDelay = configManager.getMaxBatchDelay();
        int size = configManager.getMaxBatchSize() - 1;
        long begin = System.currentTimeMillis();
        for (int i = 0; i < size; ++i) {
            if (job.isControlCmd()) {
                jobs.remove(job.getJobId());

                ListIterator<Job> iterator =
                        new ArrayList<>(jobs.values()).listIterator(jobs.size());
                while (iterator.hasNext()) {
                    Job j = iterator.next();
                    model.addFirst(j);
                }
                ModelLoadRequest req = new ModelLoadRequest(model.getModelName());
                req.setModelPath(model.getModelDir());
                return req;
            }

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

        ModelInferenceRequest req = new ModelInferenceRequest(model.getModelName());
        for (Job j : jobs.values()) {
            RequestBatch batch = new RequestBatch();
            batch.setRequestId(j.getJobId());
            batch.appendModelInput(
                    new ModelInputs(
                            "base64",
                            //Base64.getMimeEncoder().encodeToString(j.getPayload().getData()),
                            Base64.getEncoder().encodeToString(j.getPayload().getData()),
                            "data"));
            req.appendRequestBatches(batch);
        }
        return req;
    }

    public void sendResponse(ModelWorkerResponse message) {
        // TODO: Handle prediction level code

        if (message.getCode().equals(String.valueOf(200))) {
            if (jobs.isEmpty()) {
                // this is from initial load.
                return;
            }

            for (Predictions payload : message.getPredictions()) {
                String jobId = payload.getRequestId();
                Job job = jobs.remove(jobId);
                if (job == null) {
                    throw new IllegalStateException("Unexpected job: " + jobId);
                }
                job.response(
                        Base64.getDecoder().decode(payload.getValue()),
                        model.getResponseContentType());
            }
        } else {
            for (String reqId : jobs.keySet()) {
                Job j = jobs.remove(reqId);
                if (j == null) {
                    throw new IllegalStateException("Unexpected job: " + reqId);
                }
                String err =
                        "code"
                                + ":"
                                + message.getCode()
                                + ","
                                + "message"
                                + ":"
                                + message.getMessage();
                j.response(err.getBytes(Charset.forName("UTF-8")), "application/json");
            }
            if (!jobs.isEmpty()) {
                throw new IllegalStateException("Not all jobs get response.");
            }
        }
    }

    public void sendError(Object message, String error) {
        //        byte[] body = error.getBytes(StandardCharsets.UTF_8);
        //        for (Payload payload : message.getPayloads()) {
        //            String jobId = payload.getId();
        //            Job job = jobs.remove(jobId);
        //            if (job == null) {
        //                throw new IllegalStateException("Unexpected job: " + jobId);
        //            }
        //            job.response(body, HttpHeaderValues.APPLICATION_JSON);
        //        }
        //        if (!jobs.isEmpty()) {
        //            throw new IllegalStateException("Not all jobs get response.");
        //        }
    }
}
