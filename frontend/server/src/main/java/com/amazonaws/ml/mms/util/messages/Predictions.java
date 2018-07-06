package com.amazonaws.ml.mms.util.messages;

public class Predictions {
    private String requestId;
    private String value; // base64 encoded

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
