package com.amazonaws.ml.mms.openapi;

import java.util.Map;

public class Response {

    private transient String code;
    private String description;
    private Property schema;
    private Map<String, Object> examples;
    private Map<String, Property> headers;

    public Response() {}

    public Response(String code, String description) {
        this.code = code;
        this.description = description;
    }

    public String getCode() {
        return code;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Property getSchema() {
        return schema;
    }

    public void setSchema(Property schema) {
        this.schema = schema;
    }

    public Map<String, Object> getExamples() {
        return examples;
    }

    public void setExamples(Map<String, Object> examples) {
        this.examples = examples;
    }

    public Map<String, Property> getHeaders() {
        return headers;
    }

    public void setHeaders(Map<String, Property> headers) {
        this.headers = headers;
    }
}
