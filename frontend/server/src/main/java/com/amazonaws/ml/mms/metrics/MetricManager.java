package com.amazonaws.ml.mms.metrics;

import com.amazonaws.ml.mms.util.ConfigManager;
import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Type;

import java.io.IOException;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Map;

public class MetricManager {
    public Timer t;
    public TimerTask timerTask;
    public MetricStore metricStore = new MetricStore();
    public Gson gson;
    public String metricJsonString;
    private Type metricType = new TypeToken<Map<String, Map<String,Metric>>>() {
    }.getType();
    private Logger logger = LoggerFactory.getLogger(MetricManager.class);
    private Logger metrics_logger = LoggerFactory.getLogger(ConfigManager.MMS_METRICS_LOGGER);
    public MetricManager(int timeInterval) {
        this.timerTask = new TimerTask() {

            @Override
            public void run() {
                // Read and convert
                MetricCollector collector = new MetricCollector();
                try {
                    collector.collect();
                    MetricManager.this.metricJsonString = collector.jsonString;
                    MetricManager.this.metricStore.map = MetricManager.this.gson.fromJson(collector.jsonString, metricType);
                    metrics_logger.info(MetricManager.this.metricJsonString);
                }
                catch(IOException | JsonParseException e) {
                    logger.error(e.getMessage());
                }

            }
        };
        this.t = new Timer("Metrics Timer");
        this.gson = new Gson();
        // Every 60 seconds emit system level metrics
        //TODO: Replace time with value in config manager
        this.t.scheduleAtFixedRate(this.timerTask, 0, timeInterval);
    }
    public MetricManager(){
        this(60000);
    }
}
