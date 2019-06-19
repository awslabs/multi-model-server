package com.amazonaws.ml.mms.servingsdk_impl;

import java.lang.annotation.Annotation;
import java.util.HashMap;
import java.util.ServiceLoader;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;
import software.amazon.ai.mms.servingsdk.annotations.Endpoint;
import software.amazon.ai.mms.servingsdk.annotations.helpers.EndpointTypes;

public final class PluginLoader {

    private static PluginLoader instance = new PluginLoader();

    private PluginLoader() {}

    public static PluginLoader getInstance() {
        return instance;
    }

    private boolean validEndpoint(Annotation a, EndpointTypes type) {
        return a instanceof Endpoint
                && !((Endpoint) a).urlPattern().isEmpty()
                && ((Endpoint) a).endpointType().equals(type);
    }

    private HashMap<String, ModelServerEndpoint> getEndpoints(EndpointTypes type) {
        ServiceLoader<ModelServerEndpoint> loader = ServiceLoader.load(ModelServerEndpoint.class);
        HashMap<String, ModelServerEndpoint> ep = new HashMap<>();
        for (ModelServerEndpoint mep : loader) {
            Class<? extends ModelServerEndpoint> modelServerEndpointClassObj = mep.getClass();
            Annotation[] annotations = modelServerEndpointClassObj.getAnnotations();
            for (Annotation a : annotations) {
                if (validEndpoint(a, type)) {
                    ep.put(((Endpoint) a).urlPattern(), mep);
                }
            }
        }
        return ep;
    }

    public HashMap<String, ModelServerEndpoint> getAllInferenceServingEndpoints() {
        return getEndpoints(EndpointTypes.INFERENCE);
    }

    public HashMap<String, ModelServerEndpoint> getAllManagementServingEndpoints() {
        return getEndpoints(EndpointTypes.MANAGEMENT);
    }
}
