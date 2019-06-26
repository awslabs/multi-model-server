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
package com.amazonaws.ml.mms;

import com.amazonaws.ml.mms.http.InferenceRequestHandler;
import com.amazonaws.ml.mms.http.ManagementRequestHandler;
import com.amazonaws.ml.mms.servingsdk.impl.PluginsManager;
import com.amazonaws.ml.mms.util.ConfigManager;
import io.netty.channel.Channel;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.ssl.SslContext;

/**
 * A special {@link io.netty.channel.ChannelInboundHandler} which offers an easy way to initialize a
 * {@link io.netty.channel.Channel} once it was registered to its {@link
 * io.netty.channel.EventLoop}.
 */
public class ServerInitializer extends ChannelInitializer<Channel> {

    private final boolean managementServer;
    private SslContext sslCtx;

    /**
     * Creates a new {@code HttpRequestHandler} instance.
     *
     * @param sslCtx null if SSL is not enabled
     * @param managementServer true to initialize a management server instead of an API Server
     */
    public ServerInitializer(SslContext sslCtx, boolean managementServer) {
        this.sslCtx = sslCtx;
        this.managementServer = managementServer;
    }

    /** {@inheritDoc} */
    @Override
    public void initChannel(Channel ch) {
        ChannelPipeline pipeline = ch.pipeline();
        int maxRequestSize = ConfigManager.getInstance().getMaxRequestSize();
        if (sslCtx != null) {
            pipeline.addLast("ssl", sslCtx.newHandler(ch.alloc()));
        }
        pipeline.addLast("http", new HttpServerCodec());
        pipeline.addLast("aggregator", new HttpObjectAggregator(maxRequestSize));
        if (managementServer) {
            pipeline.addLast(
                    "handler",
                    new ManagementRequestHandler(
                            PluginsManager.getInstance().getManagementEndpoints()));
        } else {
            pipeline.addLast(
                    "handler",
                    new InferenceRequestHandler(
                            PluginsManager.getInstance().getInferenceEndpoints()));
        }
    }
}
