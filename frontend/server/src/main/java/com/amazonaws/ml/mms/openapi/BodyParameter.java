package com.amazonaws.ml.mms.openapi;

public class BodyParameter extends Parameter {

    private Schema schema;

    public BodyParameter() {
        this(null, null);
    }

    public BodyParameter(String name, Schema schema) {
        in = "body";
        this.name = name;
        this.schema = schema;
    }

    public Schema getSchema() {
        return schema;
    }

    public void setSchema(Schema schema) {
        this.schema = schema;
    }
}
