/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.archive;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Signature {

    private Map<String, List<Parameter>> request;
    private Map<String, List<Parameter>> response;

    public Signature() {
        request = new LinkedHashMap<>();
        response = new LinkedHashMap<>();
    }

    public Map<String, List<Parameter>> getRequest() {
        return request;
    }

    public void setRequest(Map<String, List<Parameter>> request) {
        this.request = request;
    }

    public Map<String, List<Parameter>> getResponse() {
        return response;
    }

    public void setResponse(Map<String, List<Parameter>> response) {
        this.response = response;
    }

    public void addRequest(String contentType, Parameter parameter) {
        List<Parameter> parameters = request.computeIfAbsent(contentType, k -> new ArrayList<>());
        if (parameter != null) {
            parameters.add(parameter);
        }
    }

    public void addResponse(String contentType, Parameter parameter) {
        List<Parameter> parameters = response.computeIfAbsent(contentType, k -> new ArrayList<>());
        if (parameter != null) {
            parameters.add(parameter);
        }
    }

    public static final class Parameter {

        private String name;
        private Boolean required;
        private String type;
        private String description;
        private int[] shape;
        private String contentType;

        public Parameter() {}

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public boolean isRequired() {
            return required != null && required;
        }

        public Boolean getRequired() {
            return required;
        }

        public void setRequired(Boolean required) {
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
