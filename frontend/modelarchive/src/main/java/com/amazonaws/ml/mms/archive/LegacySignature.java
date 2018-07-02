package com.amazonaws.ml.mms.archive;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class LegacySignature {

    private List<LegacyShape> inputs;
    private List<LegacyShape> outputs;

    @SerializedName("input_type")
    private String inputContentType;

    @SerializedName("output_type")
    private String outputContentType;

    public LegacySignature() {}

    public List<LegacyShape> getInputs() {
        return inputs;
    }

    public void setInputs(List<LegacyShape> inputs) {
        this.inputs = inputs;
    }

    public List<LegacyShape> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<LegacyShape> outputs) {
        this.outputs = outputs;
    }

    public String getInputContentType() {
        return inputContentType;
    }

    public void setInputContentType(String inputContentType) {
        this.inputContentType = inputContentType;
    }

    public String getOutputContentType() {
        return outputContentType;
    }

    public void setOutputContentType(String outputContentType) {
        this.outputContentType = outputContentType;
    }

    public Signature migrate() {
        Signature signature = new Signature();
        // MMS 0.4 only support "image/jpeg" and "application/json"
        for (LegacyShape legacyShape : inputs) {
            Signature.Parameter parameter = new Signature.Parameter();
            parameter.setName(legacyShape.getName());
            parameter.setShape(legacyShape.getShape());
            signature.addRequest(inputContentType, parameter);
        }

        // MMS 0.4 actually only output json data
        signature.addResponse("application/json", null);
        return signature;
    }

    public static final class LegacyShape {

        @SerializedName("data_name")
        private String name;

        @SerializedName("data_shape")
        private int[] shape;

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public int[] getShape() {
            return shape;
        }

        public void setShape(int[] shape) {
            this.shape = shape;
        }
    }
}
