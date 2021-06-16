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
import com.amazonaws.ml.mms.util.messages.RequestInput;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BatchAggregator {
    static final Logger logger = LoggerFactory.getLogger(BatchAggregator.class);
    Model model;
    ArrayBlockingQueue<BaseModelRequest> reqQ;
    ArrayBlockingQueue<Map<String, Job>> jobQ;
    String threadName;
    private ExecutorService batchHandlerService;

    public BatchAggregator(Model model) {
        this.model = model;
        // this.lastJobs = new LinkedHashMap<>();
        reqQ = new ArrayBlockingQueue<>(1);
        jobQ = new ArrayBlockingQueue<>(2);
    }

    public void setThreadName(String threadName) {
        logger.info("set threadName=" + threadName);
        this.threadName = threadName;
    }

    public String getThreadName() {
        return threadName;
    }

    public void startBatchHandlerService(String threadName) {
        setThreadName(threadName);
        if (batchHandlerService == null) {
            batchHandlerService = Executors.newSingleThreadExecutor();
            batchHandlerService.execute(new BatchHandler());
        }
    }

    public void stopBatchHandlerService() {
        if (batchHandlerService != null) {
            batchHandlerService.shutdown();
        }
        batchHandlerService = null;
    }

    public BaseModelRequest getRequest(WorkerState state) throws InterruptedException {
        return reqQ.take();
        // lastJobs = jobQ.peek();
        // return req;
    }

    public void sendResponse(ModelWorkerResponse message) {
        // TODO: Handle prediction level code
        Map<String, Job> jobs = jobQ.poll();
        if (message.getCode() == 200) {
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
                        prediction.getResp(),
                        prediction.getContentType(),
                        prediction.getStatusCode(),
                        prediction.getReasonPhrase(),
                        prediction.getHeaders());
            }
        } else {
            for (String reqId : jobs.keySet()) {
                Job j = jobs.remove(reqId);
                if (j == null) {
                    throw new IllegalStateException("Unexpected job: " + reqId);
                }
                j.sendError(HttpResponseStatus.valueOf(message.getCode()), message.getMessage());
            }
            if (!jobs.isEmpty()) {
                throw new IllegalStateException("Not all jobs get response.");
            }
        }
    }

    public void sendError(BaseModelRequest message, String error, HttpResponseStatus status) {
        if (message instanceof ModelLoadModelRequest) {
            logger.warn("Load model failed: {}, error: {}", message.getModelName(), error);
            return;
        }

        Map<String, Job> jobs = jobQ.poll();
        if (message != null) {
            ModelInferenceRequest msg = (ModelInferenceRequest) message;
            for (RequestInput req : msg.getRequestBatch()) {
                String requestId = req.getRequestId();
                Job job = jobs.remove(requestId);
                if (job == null) {
                    logger.error("Unexpected job: " + requestId);
                } else {
                    job.sendError(status, error);
                }
            }
            if (!jobs.isEmpty()) {
                jobs.clear();
                logger.error("Not all jobs get response.");
            }
        } else {
            // Send the error message to all the jobs
            if (jobs != null) {
                for (Map.Entry<String, Job> j : jobs.entrySet()) {
                    String jobsId = j.getValue().getJobId();
                    Job job = jobs.remove(jobsId);

                    if (job.isControlCmd()) {
                        job.sendError(status, error);
                    } else {
                        // Data message can be handled by other workers.
                        // If batch has gone past its batch max delay timer?
                        model.addFirst(job);
                    }
                }
            }
        }
    }

    private class BatchHandler implements Runnable {
        @Override
        public void run() {
            while (true) {
                Map<String, Job> jobs = new LinkedHashMap<>();
                ModelInferenceRequest req = new ModelInferenceRequest(model.getModelName());
                boolean loadModelJob = false;

                try {
                    model.pollBatch(threadName, jobs);
                    if (!jobs.isEmpty()) {
                        jobQ.put(jobs);

                        for (Job j : jobs.values()) {
                            if (j.isControlCmd()) {
                                if (jobs.size() > 1) {
                                    throw new IllegalStateException(
                                            "Received more than 1 control command. "
                                                    + "Control messages should be processed/retrieved one at a time.");
                                }
                                RequestInput input = j.getPayload();
                                int gpuId = -1;
                                String gpu = input.getStringParameter("gpu");
                                if (gpu != null) {
                                    gpuId = Integer.parseInt(gpu);
                                }
                                reqQ.put(new ModelLoadModelRequest(model, gpuId, threadName));
                                loadModelJob = true;
                                break;
                            } else {
                                j.setScheduled();
                                req.addRequest(j.getPayload());
                            }
                        }
                        if (!loadModelJob) {
                            reqQ.put(req);
                        }
                    }
                } catch (InterruptedException e) {
                    logger.debug("Aggregator for " + threadName + " got interrupted.", e);
                    break;
                } catch (IllegalArgumentException e) {
                    logger.debug("Aggregator for " + threadName + " got illegal argument.", e);
                }
            }
        }
    }
}
