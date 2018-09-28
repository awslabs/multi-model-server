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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerStateListener {

    private static final Logger logger = LoggerFactory.getLogger(WorkerStateListener.class);
    private CompletableFuture<Boolean> future;
    private AtomicInteger count;

    public WorkerStateListener(CompletableFuture<Boolean> future, int count) {
        this.future = future;
        this.count = new AtomicInteger(count);
    }

    public void notifyChangeState(String modelName, WorkerState state) {
        logger.info("{} worker state is: {}", modelName, state);
        // Update success and fail counts
        if (state == WorkerState.WORKER_MODEL_LOADED) {
            if (count.decrementAndGet() == 0) {
                future.complete(Boolean.TRUE);
            }
        }
        if (state == WorkerState.WORKER_ERROR || state == WorkerState.WORKER_STOPPED) {
            future.complete(Boolean.FALSE);
        }
    }
}
