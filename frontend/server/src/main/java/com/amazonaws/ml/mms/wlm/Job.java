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

import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import com.amazonaws.ml.mms.util.messages.WorkerCommands;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Job {

    private static final Logger logger = LoggerFactory.getLogger(Job.class);

    private ChannelHandlerContext ctx;

    private String modelName;
    private WorkerCommands cmd; // Else its data msg or inf requests
    private RequestBatch input;
    private long begin;

    public Job(
            ChannelHandlerContext ctx, String modelName, WorkerCommands cmd, RequestBatch input) {
        this.ctx = ctx;
        this.modelName = modelName;
        this.cmd = cmd;
        this.input = input;

        begin = System.currentTimeMillis();
    }

    public String getJobId() {
        return input.getRequestId();
    }

    public String getModelName() {
        return modelName;
    }

    public WorkerCommands getCmd() {
        return cmd;
    }

    public boolean isControlCmd() {
        return !WorkerCommands.PREDICT.equals(cmd);
    }

    public RequestBatch getPayload() {
        return input;
    }

    public void response(byte[] body, CharSequence contentType) {
        FullHttpResponse resp =
                new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK);
        if (contentType != null) {
            resp.headers().set(HttpHeaderNames.CONTENT_TYPE, contentType);
        }
        resp.content().writeBytes(body);

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendHttpResponse(ctx, resp, true);
        }

        logger.debug("Inference time: {}", System.currentTimeMillis() - begin);
    }

    public void sendError(String error) {
        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, error);
        }

        logger.debug("Inference time: {}", System.currentTimeMillis() - begin);
    }
}
