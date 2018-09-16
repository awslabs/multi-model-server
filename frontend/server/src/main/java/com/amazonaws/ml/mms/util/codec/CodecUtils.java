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

import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.CorruptedFrameException;
import java.nio.charset.StandardCharsets;

public final class CodecUtils {

    public static final int END = -1;
    public static final int BUFFER_UNDER_RUN = -3;

    private CodecUtils() {}

    public static int readLength(ByteBuf byteBuf, int maxLength) {
        int size = byteBuf.readableBytes();
        if (size < 4) {
            return BUFFER_UNDER_RUN;
        }

        int len = byteBuf.readInt();
        if (len > maxLength) {
            throw new CorruptedFrameException("Message size exceed limit: " + len);
        }
        if (len > byteBuf.readableBytes()) {
            return BUFFER_UNDER_RUN;
        }
        return len;
    }

    public static String readString(ByteBuf byteBuf, int len) {
        return new String(read(byteBuf, len), StandardCharsets.UTF_8);
    }

    public static byte[] read(ByteBuf in, int len) {
        if (len < 0) {
            throw new CorruptedFrameException("Invalid message size: " + len);
        }

        byte[] buf = new byte[len];
        in.readBytes(buf);
        return buf;
    }
}
