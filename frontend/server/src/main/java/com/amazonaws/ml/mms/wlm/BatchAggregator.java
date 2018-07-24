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

import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelInferenceRequest;
import com.amazonaws.ml.mms.util.messages.ModelLoadModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.Predictions;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import java.nio.charset.Charset;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(BatchAggregator.class);

    private Model model;
    private Map<String, Job> jobs;

    public BatchAggregator(Model model) {
        this.model = model;
        jobs = new LinkedHashMap<>();
    }

    public BaseModelRequest getRequest(String threadName) throws InterruptedException {
        jobs.clear();

        // first job is a blocking call;
        Job job = model.nextJob(threadName);
        if (job.isControlCmd()) {
            RequestBatch input = job.getPayload();
            String gpu = input.getStringParameter("gpu");
            return new ModelLoadModelRequest(model, gpu);
        }

        jobs.put(job.getJobId(), job);

        logger.debug("get first job: {}", job.getJobId());

        long maxBatchDelay = model.getMaxBatchDelay();
        int size = model.getBatchSize() - 1;
        long begin = System.currentTimeMillis();
        for (int i = 0; i < size; ++i) {
            job = model.nextJob(threadName, maxBatchDelay);
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
            req.addRequestBatches(j.getPayload());
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

            for (Predictions prediction : message.getPredictions()) {
                String jobId = prediction.getRequestId();
                Job job = jobs.remove(jobId);
                if (job == null) {
                    throw new IllegalStateException("Unexpected job: " + jobId);
                }
                job.response(
                        Base64.getDecoder().decode(prediction.getValue()),
                        prediction.getContentType());
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

    public void sendError(BaseModelRequest message, String error) {
        if (message instanceof ModelLoadModelRequest) {
            logger.warn("Load model failed: {}", message.getModelName());
            return;
        }

        if (message != null) {
            ModelInferenceRequest msg = (ModelInferenceRequest) message;
            for (RequestBatch req : msg.getRequestBatch()) {
                String requestId = req.getRequestId();
                Job job = jobs.remove(requestId);
                if (job == null) {
                    throw new IllegalStateException("Unexpected job: " + requestId);
                }
                job.sendError(error);
            }
            if (!jobs.isEmpty()) {
                jobs.clear();
                throw new IllegalStateException("Not all jobs get response.");
            }
        } else {
            // Send the error message to all the jobs
            for (Map.Entry<String, Job> j : jobs.entrySet()) {
                String jobsId = j.getValue().getJobId();
                Job job = jobs.remove(jobsId);

                if (job.isControlCmd()) {
                    job.sendError(error);
                } else {
                    // Data message can be handled by other workers.
                    // If batch has gone past its batch max delay timer?
                    model.addFirst(job);
                }
            }
        }
    }
}
