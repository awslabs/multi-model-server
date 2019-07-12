package com.amazonaws.ml.mms.http;

import com.amazonaws.ml.mms.archive.ModelException;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

public class InvalidRequestHandler extends HttpRequestHandler {
    public InvalidRequestHandler() {}

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        throw new ResourceNotFoundException();
    }
}
