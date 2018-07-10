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

import com.amazonaws.ml.mms.http.HttpRequestHandler;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.socket.SocketChannel;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.ssl.SslContext;

/**
 * A special {@link io.netty.channel.ChannelInboundHandler} which offers an easy way to initialize a
 * {@link io.netty.channel.Channel} once it was registered to its {@link
 * io.netty.channel.EventLoop}.
 */
public class ServerInitializer extends ChannelInitializer<SocketChannel> {

    private SslContext sslCtx;

    /**
     * Creates a new {@code HttpRequestHandler} instance.
     *
     * @param sslCtx null if SSL is not enabled
     */
    public ServerInitializer(SslContext sslCtx) {
        this.sslCtx = sslCtx;
    }

    /** {@inheritDoc} */
    @Override
    public void initChannel(SocketChannel ch) {
        ChannelPipeline pipeline = ch.pipeline();
        if (sslCtx != null) {
            pipeline.addLast("ssl", sslCtx.newHandler(ch.alloc()));
        }
        pipeline.addLast("http", new HttpServerCodec());
        pipeline.addLast("aggregator", new HttpObjectAggregator(6553600));
        pipeline.addLast("handler", new HttpRequestHandler());
    }
}
