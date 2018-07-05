package com.amazonaws.ml.mms.util.messages;

public class ModelLoadRequest extends AbstractRequest {

    /**
     * ModelLoadRequest is a interface between frontend and backend to notify the backend to load a
     * particular model.
     */
    private String modelPath;

    private String gpu;

    public ModelLoadRequest(String modelName) {
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
