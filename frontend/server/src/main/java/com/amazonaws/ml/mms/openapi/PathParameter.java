package com.amazonaws.ml.mms.openapi;

public class PathParameter extends Parameter {

    public PathParameter() {
        this(null, null);
    }

    public PathParameter(String name, String description) {
        this.name = name;
        this.description = description;
        in = "path";
        required = true;
    }
}
