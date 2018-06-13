package com.amazonaws.ml.mms.wlm;

import java.util.ArrayList;
import java.util.List;

public class Message {

    private String modelName;

    private List<Payload> payloads;

    public Message(String modelName) {
        this.modelName = modelName;
        payloads = new ArrayList<>();
    }

    public String getModelName() {
        return modelName;
    }

    public void addPayload(Payload payload) {
        payloads.add(payload);
    }

    public List<Payload> getPayloads() {
        return payloads;
    }
}
