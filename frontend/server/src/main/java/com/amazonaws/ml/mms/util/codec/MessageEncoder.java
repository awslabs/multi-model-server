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
