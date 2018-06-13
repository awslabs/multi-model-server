package com.amazonaws.ml.mms.archive;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Signature {

    private Request request;
    private Response response;

    public Signature() {}

    public Request getRequest() {
        return request;
    }

    public void setRequest(Request request) {
        this.request = request;
    }

    public Response getResponse() {
        return response;
    }

    public void setResponse(Response response) {
        this.response = response;
    }

    public static final class Request {

        private String contentType;
        private List<Shape> inputs;

        public String getContentType() {
            return contentType;
        }

        public void setContentType(String contentType) {
            this.contentType = contentType;
        }

        public List<Shape> getInputs() {
            return inputs == null ? Collections.emptyList() : inputs;
        }

        public void setInputs(List<Shape> inputs) {
            this.inputs = inputs;
        }

        public void addInputShape(Shape shape) {
            if (inputs == null) {
                inputs = new ArrayList<>();
            }
            inputs.add(shape);
        }
    }

    public static final class Response {

        private String contentType;
        private List<Shape> outputs;

        public String getContentType() {
            return contentType;
        }

        public void setContentType(String contentType) {
            this.contentType = contentType;
        }

        public List<Shape> getOutputs() {
            return outputs == null ? Collections.emptyList() : outputs;
        }

        public void setOutputs(List<Shape> outputs) {
            this.outputs = outputs;
        }

        public void addOutputShape(Shape shape) {
            if (outputs == null) {
                outputs = new ArrayList<>();
            }
            outputs.add(shape);
        }
    }

    public static final class Shape {

        private String name;
        private boolean required;
        private String type;
        private String description;
        private int[] shape;
        private String contentType;

        public Shape() {}

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public boolean isRequired() {
            return required;
        }

        public void setRequired(boolean required) {
            this.required = required;
        }

        public String getType() {
            return type == null ? "string" : type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public int[] getShape() {
            return shape;
        }

        public void setShape(int[] shape) {
            this.shape = shape;
        }

        public String getContentType() {
            return contentType;
        }

        public void setContentType(String contentType) {
            this.contentType = contentType;
        }
    }
}
