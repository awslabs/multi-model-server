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

import java.nio.charset.StandardCharsets;

public class Payload {

    private String id;
    private byte[] data;

    public Payload() {}

    public Payload(String id, String data) {
        this.id = id;
        if (data != null) {
            this.data = data.getBytes(StandardCharsets.UTF_8);
        }
    }

    public Payload(String id, byte[] data) {
        this.id = id;
        this.data = data;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public byte[] getData() {
        return data;
    }

    public void setData(String data) {
        this.data = data.getBytes(StandardCharsets.UTF_8);
    }

    public void setData(byte[] data) {
        this.data = data;
    }
}
