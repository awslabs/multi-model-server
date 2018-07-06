package com.amazonaws.ml.mms.util.messages;

import java.util.ArrayList;
import java.util.List;

public class ModelWorkerResponse {
    private String code;
    private String message;
    private List<Predictions> predictions;

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public List<Predictions> getPredictions() {
        return predictions;
    }

    public void setPredictions(ArrayList<Predictions> predictions) {
        this.predictions = predictions;
    }
}
