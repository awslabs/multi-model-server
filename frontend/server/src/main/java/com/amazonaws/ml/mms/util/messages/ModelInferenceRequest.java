package com.amazonaws.ml.mms.util.messages;

import java.util.ArrayList;

public class ModelInferenceRequest extends AbstractRequest {

    private String contentType;
    private ArrayList<RequestBatch> requestBatch;

    public ModelInferenceRequest(String modelName) {
        super("predict", modelName);
        requestBatch = new ArrayList<>();
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public ArrayList<RequestBatch> getRequestBatch() {
        return requestBatch;
    }

    public void setRequestBatch(ArrayList<RequestBatch> requestBatch) {
        this.requestBatch = requestBatch;
    }

    public void appendRequestBatches(RequestBatch req) {
        if (req != null) {
            this.requestBatch.add(req);
        }
    }
}
