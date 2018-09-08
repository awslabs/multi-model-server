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

import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelInferenceRequest;
import com.amazonaws.ml.mms.util.messages.ModelInputs;
import com.amazonaws.ml.mms.util.messages.ModelLoadModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelWorkerResponse;
import com.amazonaws.ml.mms.util.messages.Predictions;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageCodec;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class MessageCodec extends ByteToMessageCodec<BaseModelRequest> {

    private void encodeRequestBatch(RequestBatch req, ByteBuf out) {
        byte[] reqIdEnc = req.getRequestId().getBytes(StandardCharsets.UTF_8);
        out.writeInt(reqIdEnc.length);
        out.writeBytes(reqIdEnc);

        if (req.getContentType() != null) {
            byte[] contentTypeEnc = req.getContentType().getBytes(StandardCharsets.UTF_8);
            out.writeInt(contentTypeEnc.length);
            out.writeBytes(contentTypeEnc);
        } else {
            out.writeInt(0); // Length 0
        }

        out.writeInt(-1); // Start of List
        for (ModelInputs input : req.getModelInputs()) {
            encodeModelInputs(input, out);
        }
        out.writeInt(-2); // End of List
    }

    private void encodeModelInputs(ModelInputs modelInputs, ByteBuf out) {
        byte[] modelInputName = modelInputs.getName().getBytes(StandardCharsets.UTF_8);
        out.writeInt(modelInputName.length);
        out.writeBytes(modelInputName);

        if (modelInputs.getContentType() != null) {
            byte[] contentTypeEnc = modelInputs.getContentType().getBytes(StandardCharsets.UTF_8);
            out.writeInt(contentTypeEnc.length);
            out.writeBytes(contentTypeEnc);
        } else {
            out.writeInt(0); // Length 0
        }

        out.writeInt(modelInputs.getValue().length);
        out.writeBytes(modelInputs.getValue());
    }

    @Override
    protected void encode(ChannelHandlerContext ctx, BaseModelRequest msg, ByteBuf out) {
        if (msg instanceof ModelLoadModelRequest) {
            out.writeInt(1); // SOM
            out.writeInt(1); // load 1

            byte[] modelNameEnc = msg.getModelName().getBytes(StandardCharsets.UTF_8);
            out.writeInt(modelNameEnc.length);
            out.writeBytes(modelNameEnc);

            byte[] modelPathEnc =
                    ((ModelLoadModelRequest) msg).getModelPath().getBytes(StandardCharsets.UTF_8);
            out.writeInt(modelPathEnc.length);
            out.writeBytes(modelPathEnc);

            if (((ModelLoadModelRequest) msg).getBatchSize() >= 0) {
                out.writeInt(((ModelLoadModelRequest) msg).getBatchSize());
            } else {
                out.writeInt(1);
            }

            byte[] handlerEnc =
                    ((ModelLoadModelRequest) msg).getHandler().getBytes(StandardCharsets.UTF_8);
            out.writeInt(handlerEnc.length);
            out.writeBytes(handlerEnc);

            if (((ModelLoadModelRequest) msg).getGpu() != null) {
                out.writeInt(Integer.parseInt(((ModelLoadModelRequest) msg).getGpu()));
            } else {
                out.writeInt(-1);
            }
        } else if (msg instanceof ModelInferenceRequest) {
            out.writeInt(1);
            out.writeInt(2); // Predict/inference: 2

            byte[] modelNameEnc = msg.getModelName().getBytes(StandardCharsets.UTF_8);
            out.writeInt(modelNameEnc.length);
            out.writeBytes(modelNameEnc);

            out.writeInt(-1); // Start of List
            for (RequestBatch batch : ((ModelInferenceRequest) msg).getRequestBatch()) {
                encodeRequestBatch(batch, out);
            }
            out.writeInt(-2); // End of List
        }
    }

    private int decodeGetInt(ByteBuf in) {
        if (in.readableBytes() >= 4) {
            return in.readInt();
        }
        throw new HttpPostRequestDecoder.NotEnoughDataDecoderException("Not enough data");
    }

    private String decodeGetString(ByteBuf in, int len) {
        if (in.readableBytes() >= len) {
            return in.readCharSequence(len, StandardCharsets.UTF_8).toString();
        }
        throw new HttpPostRequestDecoder.NotEnoughDataDecoderException("Not enough data");
    }

    private void decodeByteBuff(ByteBuf in, byte[] dest, int len) {
        if ((len < 0) || (in.readableBytes() < len)) {
            throw new HttpPostRequestDecoder.NotEnoughDataDecoderException("Not enough data");
        }
        in.readBytes(dest, 0, len);
    }

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {

        ModelWorkerResponse response = new ModelWorkerResponse();
        try {
            response.setCode(Integer.toString(decodeGetInt(in)));

            int length = decodeGetInt(in);
            response.setMessage(decodeGetString(in, length));

            length = decodeGetInt(in);
            List<Predictions> predictionsList = new ArrayList<>();
            if (length < 0) {
                // There are a list of predictions
                while (length != -2) {
                    Predictions p = new Predictions();
                    length = decodeGetInt(in);
                    if (length < 0) {
                        continue;
                    }

                    p.setRequestId(decodeGetString(in, length));

                    length = decodeGetInt(in);
                    if (length < 0) {
                        continue;
                    }

                    p.setContentType(decodeGetString(in, length));

                    length = decodeGetInt(in);
                    if (length < 0) {
                        continue;
                    }
                    p.setResp(new byte[length]);
                    decodeByteBuff(in, p.getResp(), length);
                    predictionsList.add(p);
                }
                response.setPredictions(predictionsList);
            }

            out.add(response);
        } catch (HttpPostRequestDecoder.NotEnoughDataDecoderException e) {
            in.resetReaderIndex();
        }
    }
}
