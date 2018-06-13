package com.amazonaws.ml.mms.openapi;

public class IntegerProperty extends Property {

    public IntegerProperty() {
        this(null, false, null);
    }

    public IntegerProperty(String name, String description) {
        this(name, false, description);
    }

    public IntegerProperty(String name, boolean required, String description) {
        super("integer", name, required, description);
        format = "int32";
    }
}
