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
package com.amazonaws.ml.mms.util.codec;

import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;

@ChannelHandler.Sharable
public class MessageEncoder extends MessageToMessageEncoder<BaseModelRequest> {

    @Override
    protected void encode(ChannelHandlerContext ctx, BaseModelRequest msg, List<Object> out) {
        String cmd = JsonUtils.GSON.toJson(msg) + "\r\n";
        ByteBuf buf = Unpooled.copiedBuffer(cmd, StandardCharsets.UTF_8);
        out.add(buf);
    }
}
