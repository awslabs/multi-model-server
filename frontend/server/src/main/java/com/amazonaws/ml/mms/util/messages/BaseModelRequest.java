package com.amazonaws.ml.mms.util.messages;

public class BaseModelRequest {

    private String command;
    private String modelName;

    public BaseModelRequest(String command, String modelName) {
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
