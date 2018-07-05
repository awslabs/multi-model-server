package com.amazonaws.ml.mms.util.messages;

public abstract class AbstractRequest {

    private String command;
    private String modelName;

    public AbstractRequest(String command, String modelName) {
        this.command = command;
        this.modelName = modelName;
    }

    public String getCommand() {
        return command;
    }

    public String getModelName() {
        return modelName;
    }
}
