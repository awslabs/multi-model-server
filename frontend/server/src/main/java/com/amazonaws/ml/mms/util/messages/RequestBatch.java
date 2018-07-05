package com.amazonaws.ml.mms.util.messages;

import java.util.ArrayList;

public class RequestBatch {
    private String requestId;
    private String encoding;
    private ArrayList<ModelInputs> modelInputs;

    public RequestBatch() {
        modelInputs = new ArrayList<>();
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getEncoding() {
        return encoding;
    }

    public void setEncoding(String encoding) {
        this.encoding = encoding;
    }

    public ArrayList<ModelInputs> getModelInputs() {
        return modelInputs;
    }

    public void setModelInputs(ArrayList<ModelInputs> modelInputs) {
        this.modelInputs = modelInputs;
    }

    public void appendModelInput(ModelInputs modelInput) {
        if (modelInput != null) {
            this.modelInputs.add(modelInput);
        }
    }
}
