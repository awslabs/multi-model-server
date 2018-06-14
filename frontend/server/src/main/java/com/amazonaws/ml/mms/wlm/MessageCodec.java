package com.amazonaws.ml.mms.wlm;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageCodec;
import io.netty.handler.codec.CorruptedFrameException;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class MessageCodec extends ByteToMessageCodec<Message> {

    private int maxSize;

    public MessageCodec() {
        this(6553500);
    }

    public MessageCodec(int maxSize) {
        this.maxSize = maxSize;
    }

    @Override
    protected void encode(ChannelHandlerContext ctx, Message msg, ByteBuf buf) {
        List<Payload> payloads = msg.getPayloads();

        buf.writeByte('M');
        byte[] modelName = msg.getModelName().getBytes(StandardCharsets.UTF_8);
        buf.writeInt(modelName.length);
        buf.writeBytes(modelName);

        for (Payload payload : payloads) {
            String id = payload.getId();
            byte[] data = payload.getData();

            buf.writeInt(id.length());
            buf.writeBytes(id.getBytes(StandardCharsets.US_ASCII));
            buf.writeInt(data.length);
            buf.writeBytes(data);
        }
        buf.writeInt(-1);
    }

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        int size = in.readableBytes();
        if (size < 5) {
            return;
        }

        in.markReaderIndex();

        // Check the magic number.
        int magicNumber = in.readUnsignedByte();
        if (magicNumber != 'M') {
            in.resetReaderIndex();
            throw new CorruptedFrameException("Invalid magic number: " + magicNumber);
        }

        byte[] buf = read(in);
        if (buf == null) {
            in.resetReaderIndex();
            throw new CorruptedFrameException("Missing model name.");
        }

        Message message = new Message(new String(buf, StandardCharsets.UTF_8));
        while (true) {
            if (in.readableBytes() < 4) {
                in.resetReaderIndex();
                return;
            }

            int len = in.readInt();
            if (len == -1) {
                // end of message
                break;
            }

            buf = read(in, len);
            if (buf == null) {
                in.resetReaderIndex();
                return;
            }

            String id = new String(buf, StandardCharsets.US_ASCII);
            buf = read(in);
            if (buf == null) {
                in.resetReaderIndex();
                return;
            }
            Payload payload = new Payload(id, buf);
            message.addPayload(payload);
        }

        out.add(message);
    }

    private byte[] read(ByteBuf in) {
        int size = in.readableBytes();
        if (size < 4) {
            in.resetReaderIndex();
            return null;
        }

        int len = in.readInt();
        return read(in, len);
    }

    private byte[] read(ByteBuf in, int len) {
        if (in.readableBytes() < len) {
            return null;
        }

        if (len > maxSize) {
            throw new CorruptedFrameException("Message size exceed limit: " + len);
        }

        byte[] buf = new byte[len];
        in.readBytes(buf);
        return buf;
    }
}
