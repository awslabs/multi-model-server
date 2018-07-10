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
package com.amazonaws.ml.mms.wlm;

import java.util.ArrayList;
import java.util.List;

public class Message {

    private String modelName;

    private List<Payload> payloads;

    public Message(String modelName) {
        this.modelName = modelName;
        payloads = new ArrayList<>();
    }

    public String getModelName() {
        return modelName;
    }

    public void addPayload(Payload payload) {
        payloads.add(payload);
    }

    public List<Payload> getPayloads() {
        return payloads;
    }
}
