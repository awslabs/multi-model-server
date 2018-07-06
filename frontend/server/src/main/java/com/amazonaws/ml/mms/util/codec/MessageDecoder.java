package com.amazonaws.ml.mms.util.codec;

import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageDecoder;
import java.util.List;

@ChannelHandler.Sharable
public class MessageDecoder extends MessageToMessageDecoder<String> {

    @Override
    protected void decode(ChannelHandlerContext ctx, String msg, List<Object> out) {
        if (msg == null || msg.isEmpty()) {
            return;
        }
        ModelWorkerResponse resp = JsonUtils.GSON.fromJson(msg, ModelWorkerResponse.class);
        out.add(resp);
    }
}
