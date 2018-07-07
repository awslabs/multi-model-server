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

import java.util.ArrayList;

public class RequestBatch {
    private String requestId;
    private String encoding;
    private ArrayList<ModelInputs> modelInputs;

    public RequestBatch() {
        modelInputs = new ArrayList<>();
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getEncoding() {
        return encoding;
    }

    public void setEncoding(String encoding) {
        this.encoding = encoding;
    }

    public ArrayList<ModelInputs> getModelInputs() {
        return modelInputs;
    }

    public void setModelInputs(ArrayList<ModelInputs> modelInputs) {
        this.modelInputs = modelInputs;
    }

    public void appendModelInput(ModelInputs modelInput) {
        if (modelInput != null) {
            this.modelInputs.add(modelInput);
        }
    }
}
