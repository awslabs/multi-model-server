package com.amazonaws.ml.mms.openapi;

public class StringProperty extends Property {

    public StringProperty() {
        this(null, false, null);
    }

    public StringProperty(String name, String description) {
        this(name, false, description);
    }

    public StringProperty(String name, boolean required, String description) {
        super("string", name, required, description);
    }
}
