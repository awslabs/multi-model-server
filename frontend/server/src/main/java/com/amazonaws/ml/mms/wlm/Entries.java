package com.amazonaws.ml.mms.wlm;

public class Entries {
    private String encoding;
    private String data;
    private String name;

    public Entries() {}

    public Entries(String encoding, String data, String name) {
        this.encoding = encoding;
        this.data = data;
        this.name = name;
    }

    public String getEncoding() {
        return encoding;
    }

    public void setEncoding(String encoding) {
        this.encoding = encoding;
    }

    public String getData() {
        return data;
    }

    public void setData(String data) {
        this.data = data;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
