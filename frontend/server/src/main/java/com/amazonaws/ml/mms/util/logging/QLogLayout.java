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
import org.apache.logging.log4j.core.Layout;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.config.Node;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.plugins.PluginFactory;
import org.apache.logging.log4j.core.layout.AbstractStringLayout;
import org.apache.logging.log4j.message.Message;

@Plugin(
        name = "QLogLayout",
        category = Node.CATEGORY,
        elementType = Layout.ELEMENT_TYPE,
        printObject = true)
public class QLogLayout extends AbstractStringLayout {

    public QLogLayout() {
        super(null, null, null);
    }

    /**
     * Model server also supports query log formatting.
     *
     * <p>To enable Query Log format, change the layout as follows
     *
     * <pre>
     *     log4j.appender.model_metrics.layout = com.amazonaws.ml.mms.util.logging.QLogLayout
     * </pre>
     *
     * This enables logs which are shown as following
     *
     * <pre>
     *     HostName=hostName
     *     RequestId=004bd136-063c-4102-a070-d7aff5add939
     *     Marketplace=US
     *     StartTime=1542275707
     *     Program=MXNetModelServer
     *     Metrics=PredictionTime=45 Milliseconds ModelName|squeezenet  Level|Model
     *     EOE
     * </pre>
     *
     * <b>Note</b>: The following entities in this metrics can be customized.
     *
     * <ul>
     *   <li><b>Marketplace</b> : This can be customized by setting the "REALM" system environment
     *       variable.
     *   <li><b>Program</b> : This entity can be customized by setting "MXNETMODELSERVER_PROGRAM"
     *       environment variable.
     * </ul>
     *
     * Example: If the above environment variables are set to the following,
     *
     * <pre>
     *     $ env
     *     REALM=someRealm
     *     MXNETMODELSERVER_PROGRAM=someProgram
     * </pre>
     *
     * This produces the metrics as follows
     *
     * <pre>
     *    HostName=hostName
     *    RequestId=004bd136-063c-4102-a070-d7aff5add939
     *    Marketplace=someRealm
     *    StartTime=1542275707
     *    Program=someProgram
     *    Metrics=PredictionTime=45 Milliseconds ModelName|squeezenet  Level|Model
     *    EOE
     * </pre>
     *
     * @param event
     * @return
     */
    @Override
    public String toSerializable(LogEvent event) {
        Message eventMessage = event.getMessage();
        if (eventMessage == null || eventMessage.getParameters() == null) {
            return null;
        }
        String programName =
                getStringOrDefault(System.getenv("MXNETMODELSERVER_PROGRAM"), "MXNetModelServer");
        String domain = getStringOrDefault(System.getenv("DOMAIN"), "Unknown");
        long currentTimeInSec = System.currentTimeMillis() / 1000;
        Object[] parameters = eventMessage.getParameters();

        StringBuilder stringBuilder = new StringBuilder();
        for (Object obj : parameters) {
            if (obj instanceof Metric) {
                Metric metric = (Metric) obj;
                String marketPlace = System.getenv("REALM");

                stringBuilder.append("HostName=").append(metric.getHostName());

                if (metric.getRequestId() != null && !metric.getRequestId().isEmpty()) {
                    stringBuilder.append("\nRequestId=").append(metric.getRequestId());
                }

                // Marketplace format should be : <programName>:<domain>:<realm>
                if (marketPlace != null && !marketPlace.isEmpty()) {
                    stringBuilder
                            .append("\nMarketplace=")
                            .append(programName)
                            .append(':')
                            .append(domain)
                            .append(':')
                            .append(marketPlace);
                }

                stringBuilder
                        .append("\nStartTime=")
                        .append(
                                getStringOrDefault(
                                        metric.getTimestamp(), Long.toString(currentTimeInSec)));

                stringBuilder
                        .append("\nProgram=")
                        .append(programName)
                        .append("\nMetrics=")
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
            }
        }
        return stringBuilder.toString();
    }

    @PluginFactory
    public static QLogLayout createLayout() {
        return new QLogLayout();
    }

    private static String getStringOrDefault(String val, String defVal) {

        if (val == null || val.isEmpty()) {
            return defVal;
        }
        return val;
    }
}
