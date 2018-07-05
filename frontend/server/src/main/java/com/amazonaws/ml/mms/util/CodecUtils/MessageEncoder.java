package com.amazonaws.ml.mms.util.CodecUtils;

import com.amazonaws.ml.mms.util.JsonUtils;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;

@ChannelHandler.Sharable
public class MessageEncoder extends MessageToMessageEncoder<Object> {

    @Override
    protected void encode(ChannelHandlerContext ctx, Object msg, List<Object> out) {
        String cmd = JsonUtils.GSON.toJson(msg) + "\r\n";
        ByteBuf buf = Unpooled.copiedBuffer(cmd, StandardCharsets.UTF_8);
        out.add(buf);
    }
}
