package com.amazonaws.ml.mms.openapi;

public class BooleanProperty extends Property {

    public BooleanProperty() {
        this(null, false, null);
    }

    public BooleanProperty(String name, String description) {
        this(name, false, description);
    }

    public BooleanProperty(String name, boolean required, String description) {
        super("boolean", name, required, description);
    }
}
