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

import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.Predictions;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;
import java.util.ArrayList;
import java.util.List;

public class ModelResponseDecoder extends ByteToMessageDecoder {

    private static final int MAX_BUFFER_SIZE = 6553500;

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        int size = in.readableBytes();
        if (size < 9) {
            return;
        }

        in.markReaderIndex();
        boolean completed = false;
        try {
            ModelWorkerResponse resp = new ModelWorkerResponse();
            resp.setCode(in.readInt());

            int len = CodecUtils.readLength(in, MAX_BUFFER_SIZE);
            if (len == CodecUtils.BUFFER_UNDER_RUN) {
                return;
            }
            resp.setMessage(CodecUtils.readString(in, len));

            List<Predictions> predictions = new ArrayList<>();
            while ((len = CodecUtils.readLength(in, MAX_BUFFER_SIZE)) != CodecUtils.END) {
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                Predictions prediction = new Predictions();
                prediction.setRequestId(CodecUtils.readString(in, len));

                len = CodecUtils.readLength(in, MAX_BUFFER_SIZE);
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                prediction.setContentType(CodecUtils.readString(in, len));

                len = CodecUtils.readLength(in, MAX_BUFFER_SIZE);
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                prediction.setResp(CodecUtils.read(in, len));
                predictions.add(prediction);
            }
            resp.setPredictions(predictions);
            out.add(resp);
            completed = true;
        } finally {
            if (!completed) {
                in.resetReaderIndex();
            }
        }
    }
}
