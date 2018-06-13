package com.amazonaws.ml.mms.openapi;

import com.google.gson.annotations.SerializedName;

public class Property {

    protected transient String name;
    protected String type;
    protected String format;
    protected String example;
    protected transient boolean required;
    protected Integer position;
    protected String description;
    protected String title;
    protected Boolean readOnly;

    @SerializedName("default")
    protected String defaultValue;

    public Property() {}

    public Property(String type, String name, String description) {
        this(type, name, false, description);
    }

    public Property(String type, String name, boolean required, String description) {
        this.type = type;
        this.name = name;
        this.required = required;
        this.description = description;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getType() {
        return type;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public String getExample() {
        return example;
    }

    public void setExample(String example) {
        this.example = example;
    }

    public boolean isRequired() {
        return required;
    }

    public void setRequired(boolean required) {
        this.required = required;
    }

    public Integer getPosition() {
        return position;
    }

    public void setPosition(Integer position) {
        this.position = position;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public Boolean getReadOnly() {
        return readOnly;
    }

    public void setReadOnly(Boolean readOnly) {
        this.readOnly = readOnly;
    }

    public String getDefaultValue() {
        return defaultValue;
    }

    public void setDefaultValue(String defaultValue) {
        this.defaultValue = defaultValue;
    }
}
