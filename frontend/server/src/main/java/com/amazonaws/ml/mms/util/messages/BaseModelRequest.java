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

public class BaseModelRequest {

    private String command;
    private String modelName;
    private boolean isSynchronous;

    public BaseModelRequest(String command, String modelName) {
        this.command = command;
        this.modelName = modelName;
        this.isSynchronous = false;
    }

    public void setIsSynchronous() {
        isSynchronous = true;
    }

    public boolean getIsSynchronous() {
        return isSynchronous;
    }

    public String getCommand() {
        return command;
    }

    public String getModelName() {
        return modelName;
    }
}
