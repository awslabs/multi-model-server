package com.amazonaws.ml.mms.util;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public final class JsonUtils {

    public static final Gson GSON_PRETTY = new GsonBuilder().setPrettyPrinting().create();
    public static final Gson GSON = new GsonBuilder().create();

    private JsonUtils() {}
}
