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
package com.amazonaws.ml.mms.metrics;

import java.util.ArrayList;

public class Metric {
    private String name;
    private String value;
    private String unit;
    private ArrayList<Dimension> dimensions;
    private String timestamp;

    public void setName(String name) {
        this.name = name;
    }

    public void setDimensions(ArrayList<Dimension> dimensions) {
        this.dimensions = dimensions;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public String getName() {

        return name;
    }

    public ArrayList<Dimension> getDimensions() {
        return dimensions;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public Metric() {}

    public String getUnit() {
        return unit;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public void setUnit(String unit) {
        this.unit = unit;
    }
}
