package com.amazonaws.ml.mms.servingsdk_impl;

import com.amazonaws.ml.mms.util.ConfigManager;
import java.util.Properties;
import software.amazon.ai.mms.servingsdk.Context;

public class ModelServerContext implements Context {
    @Override
    public Properties getConfig() {
        return ConfigManager.getInstance().getConfiguration();
    }
}
