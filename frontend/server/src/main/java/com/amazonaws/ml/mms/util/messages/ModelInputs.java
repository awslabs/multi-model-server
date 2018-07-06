package com.amazonaws.ml.mms.util.messages;

public class ModelInputs {

    private String encoding;
    private String value;
    private String name;

    public ModelInputs(String encoding, String value, String name) {
        this.encoding = encoding;
        this.value = value;
        this.name = name;
    }

    public String getEncoding() {
        return encoding;
    }

    public void setEncoding(String encoding) {
        this.encoding = encoding;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
