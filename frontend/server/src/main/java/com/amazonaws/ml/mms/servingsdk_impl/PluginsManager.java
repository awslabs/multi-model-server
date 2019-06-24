/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazonaws.ml.mms.servingsdk_impl;

import com.amazonaws.ml.mms.http.InvalidPluginException;
import java.lang.annotation.Annotation;
import java.util.HashMap;
import java.util.ServiceLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;
import software.amazon.ai.mms.servingsdk.annotations.Endpoint;
import software.amazon.ai.mms.servingsdk.annotations.helpers.EndpointTypes;

public final class PluginsManager {

    private static PluginsManager instance = new PluginsManager();
    private Logger logger = LoggerFactory.getLogger(PluginsManager.class);

    private HashMap<String, ModelServerEndpoint> inferenceEndpoints;
    private HashMap<String, ModelServerEndpoint> managementEndpoints;

    private PluginsManager() {}

    public static PluginsManager getInstance() {
        return instance;
    }

    public void initialize() {
        inferenceEndpoints = initInferenceEndpoints();
        managementEndpoints = initManagementEndpoints();
    }

    private boolean validEndpoint(Annotation a, EndpointTypes type) {
        return a instanceof Endpoint
                && !((Endpoint) a).urlPattern().isEmpty()
                && ((Endpoint) a).endpointType().equals(type);
    }

    private HashMap<String, ModelServerEndpoint> getEndpoints(EndpointTypes type)
            throws InvalidPluginException {
        ServiceLoader<ModelServerEndpoint> loader = ServiceLoader.load(ModelServerEndpoint.class);
        HashMap<String, ModelServerEndpoint> ep = new HashMap<>();
        for (ModelServerEndpoint mep : loader) {
            Class<? extends ModelServerEndpoint> modelServerEndpointClassObj = mep.getClass();
            Annotation[] annotations = modelServerEndpointClassObj.getAnnotations();
            for (Annotation a : annotations) {
                if (validEndpoint(a, type)) {
                    if (ep.get(((Endpoint) a).urlPattern()) != null) {
                        throw new InvalidPluginException(
                                "Multiple plugins found for endpoint "
                                        + "\""
                                        + ((Endpoint) a).urlPattern()
                                        + "\"");
                    }
                    logger.info("Loading plugin for endpoint {}", ((Endpoint) a).urlPattern());
                    ep.put(((Endpoint) a).urlPattern(), mep);
                }
            }
        }
        return ep;
    }

    private HashMap<String, ModelServerEndpoint> initInferenceEndpoints() {
        return getEndpoints(EndpointTypes.INFERENCE);
    }

    private HashMap<String, ModelServerEndpoint> initManagementEndpoints() {
        return getEndpoints(EndpointTypes.MANAGEMENT);
    }

    public HashMap<String, ModelServerEndpoint> getInferenceEndpoints() {
        return inferenceEndpoints;
    }

    public HashMap<String, ModelServerEndpoint> getManagementEndpoints() {
        return managementEndpoints;
    }
}
