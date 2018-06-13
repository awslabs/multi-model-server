package com.amazonaws.ml.mms.openapi;

public class ArrayProperty extends Property {

    private Property items;

    public ArrayProperty() {
        type = "array";
    }

    public ArrayProperty(String name, String description) {
        super("array", name, description);
    }

    public ArrayProperty(String name, String description, Property items) {
        super("array", name, description);
        this.items = items;
    }

    public Property getItems() {
        return items;
    }

    public void setItems(Property items) {
        this.items = items;
    }
}
