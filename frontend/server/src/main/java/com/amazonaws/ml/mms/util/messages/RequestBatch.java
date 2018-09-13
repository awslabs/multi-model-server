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
package com.amazonaws.ml.mms.util.messages;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class RequestBatch {

    private String requestId;
    private String contentType;
    private List<ModelInputs> modelInputs;

    public RequestBatch(String requestId) {
        this.requestId = requestId;
        modelInputs = new ArrayList<>();
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public List<ModelInputs> getModelInputs() {
        return modelInputs;
    }

    public void setModelInputs(List<ModelInputs> modelInputs) {
        this.modelInputs = modelInputs;
    }

    public void addModelInput(ModelInputs modelInput) {
        modelInputs.add(modelInput);
    }

    public String getStringParameter(String key) {
        for (ModelInputs param : modelInputs) {
            if (key.equals(param.getName())) {
                return new String(param.getValue(), StandardCharsets.UTF_8);
            }
        }
        return null;
    }
}
