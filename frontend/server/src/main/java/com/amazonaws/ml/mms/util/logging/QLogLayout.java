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
package com.amazonaws.ml.mms.util.logging;

import com.amazonaws.ml.mms.metrics.Dimension;
import com.amazonaws.ml.mms.metrics.Metric;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.spi.LoggingEvent;

public class QLogLayout extends PatternLayout {

    @Override
    public String format(LoggingEvent event) {
        Object eventMessage = event.getMessage();
        if (eventMessage == null) {
            return null;
        }
        if (eventMessage instanceof Metric) {
            String marketPlace = System.getenv("REALM");
            StringBuilder stringBuilder = new StringBuilder();
            Metric metric = (Metric) eventMessage;
            stringBuilder.append("HostName=").append(metric.getHostName());
            if (metric.getRequestId() != null) {
                stringBuilder.append("\nRequestId=").append(metric.getRequestId());
            }
            if (marketPlace != null) {
                stringBuilder.append("\nMarketplace=").append(marketPlace);
            }
            stringBuilder.append("\nStartTime=").append(metric.getTimestamp());
            stringBuilder
                    .append("\nProgram=MXNetModelServer\nMetrics=")
                    .append(metric.getMetricName())
                    .append('=')
                    .append(metric.getValue())
                    .append(' ')
                    .append(metric.getUnit());
            for (Dimension dimension : metric.getDimensions()) {
                stringBuilder
                        .append(' ')
                        .append(dimension.getName())
                        .append('|')
                        .append(dimension.getValue())
                        .append(' ');
            }
            stringBuilder.append("\nEOE\n");

            return stringBuilder.toString();
        }
        return eventMessage.toString();
    }
}
