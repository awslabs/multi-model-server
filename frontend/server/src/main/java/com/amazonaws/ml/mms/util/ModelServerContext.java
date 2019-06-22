package com.amazonaws.ml.mms.util;

import com.amazonaws.ml.mms.servingsdk_impl.ModelServerModel;
import com.amazonaws.ml.mms.wlm.ModelManager;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import software.amazon.ai.mms.servingsdk.Context;
import software.amazon.ai.mms.servingsdk.Model;

public class ModelServerContext implements Context {
    @Override
    public Properties getConfig() {
        return ConfigManager.getInstance().getConfiguration();
    }

    @Override
    public Map<String, Model> getModels() {
        HashMap<String, Model> ret = new HashMap<>();
        ModelManager.getInstance()
                .getModels()
                .forEach((modelName, model) -> ret.put(modelName, new ModelServerModel(model)));
        return ret;
    }
}
