package com.amazonaws.ml.mms;

public final class TestUtils {

    private TestUtils() {}

    public static void init() {
        // set up system properties for local IDE debug
        if (System.getProperty("DEBUG") == null) {
            System.setProperty("DEBUG", "true");
        }
        if (System.getProperty("mmsConfigFile") == null) {
            System.setProperty("mmsConfigFile", "src/test/resources/config.properties");
        }
        if (System.getProperty("METRICS_LOCATION") == null) {
            System.setProperty("METRICS_LOCATION", "build/logs");
        }
        if (System.getProperty("LOG_LOCATION") == null) {
            System.setProperty("LOG_LOCATION", "build/logs");
        }
    }
}
