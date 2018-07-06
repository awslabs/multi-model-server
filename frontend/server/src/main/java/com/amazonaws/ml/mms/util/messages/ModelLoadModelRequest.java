package com.amazonaws.ml.mms.util.messages;

public class ModelLoadModelRequest extends BaseModelRequest {

    /**
     * ModelLoadModelRequest is a interface between frontend and backend to notify the backend to load a
     * particular model.
     */
    private String modelPath;

    private String gpu;

    public ModelLoadModelRequest(String modelName) {
        super("load", modelName);
    }

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public String getGpu() {
        return gpu;
    }

    public void setGpu(String gpu) {
        this.gpu = gpu;
    }
}
