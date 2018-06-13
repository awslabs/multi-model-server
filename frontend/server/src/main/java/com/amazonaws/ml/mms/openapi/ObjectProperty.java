package com.amazonaws.ml.mms.openapi;

import java.util.LinkedHashMap;
import java.util.Map;

public class ObjectProperty extends Property {

    Map<String, Property> properties;

    public ObjectProperty() {
        type = "object";
    }

    public Map<String, Property> getProperties() {
        return properties;
    }

    public void setProperties(Map<String, Property> properties) {
        this.properties = properties;
    }

    public void addProperty(Property prop) {
        if (properties == null) {
            properties = new LinkedHashMap<>();
        }
        properties.put(prop.getName(), prop);
    }
}
