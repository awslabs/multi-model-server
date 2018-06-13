package com.amazonaws.ml.mms.wlm;

import java.nio.charset.StandardCharsets;

public class Payload {

    private String id;
    private byte[] data;

    public Payload() {}

    public Payload(String id, String data) {
        this.id = id;
        if (data != null) {
            this.data = data.getBytes(StandardCharsets.UTF_8);
        }
    }

    public Payload(String id, byte[] data) {
        this.id = id;
        this.data = data;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public byte[] getData() {
        return data;
    }

    public void setData(String data) {
        this.data = data.getBytes(StandardCharsets.UTF_8);
    }

    public void setData(byte[] data) {
        this.data = data;
    }
}
