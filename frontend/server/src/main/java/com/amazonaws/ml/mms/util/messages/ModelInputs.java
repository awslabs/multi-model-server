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
import java.util.Base64;

public class ModelInputs {

    private String name;
    private String value;
    private String encoding;
    private String contentType;

    public ModelInputs() {}

    public ModelInputs(String name, String value) {
        this.name = name;
        this.value = value;
    }

    public ModelInputs(String name, byte[] data) {
        this(name, data, null);
    }

    public ModelInputs(String name, byte[] data, String contentType) {
        this.name = name;
        this.contentType = contentType;
        encoding = "base64";
        value = Base64.getEncoder().encodeToString(data);
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

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public byte[] getBytes() {
        if ("base64".equals(encoding)) {
            return Base64.getDecoder().decode(value);
        }
        return value.getBytes(StandardCharsets.UTF_8);
    }
}
