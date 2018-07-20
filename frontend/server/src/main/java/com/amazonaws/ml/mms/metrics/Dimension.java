package com.amazonaws.ml.mms.metrics;

import java.util.ArrayList;

public class Dimension {
    private String name;
    private String value;
    public Dimension(){

    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
