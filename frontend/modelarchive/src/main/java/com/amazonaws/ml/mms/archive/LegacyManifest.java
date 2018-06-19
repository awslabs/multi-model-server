package com.amazonaws.ml.mms.archive;

import com.google.gson.annotations.SerializedName;
import java.util.Map;

public class LegacyManifest {

    @SerializedName("Engine")
    private Map<String, Object> engine;

    @SerializedName("Model-Archive-Description")
    private String description;

    @SerializedName("License")
    private String license;

    @SerializedName("Model-Archive-Version")
    private String version;

    @SerializedName("Model-Server")
    private String serverVersion;

    @SerializedName("Model")
    private ModelInfo modelInfo;

    @SerializedName("Created-By")
    private CreatedBy createdBy;

    public LegacyManifest() {}

    public Map<String, Object> getEngine() {
        return engine;
    }

    public void setEngine(Map<String, Object> engine) {
        this.engine = engine;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getLicense() {
        return license;
    }

    public void setLicense(String license) {
        this.license = license;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public String getServerVersion() {
        return serverVersion;
    }

    public void setServerVersion(String serverVersion) {
        this.serverVersion = serverVersion;
    }

    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    public void setModelInfo(ModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public CreatedBy getCreatedBy() {
        return createdBy;
    }

    public void setCreatedBy(CreatedBy createdBy) {
        this.createdBy = createdBy;
    }

    public Manifest migrate() {
        Manifest manifest = new Manifest();
        manifest.setDescription(description);
        manifest.setLicense(license);

        if (createdBy != null) {
            Manifest.Publisher publisher = new Manifest.Publisher();
            publisher.setAuthor(createdBy.getAuthor());
            publisher.setEmail(createdBy.getEmail());
            manifest.setPublisher(publisher);
        }

        if (modelInfo != null) {
            Manifest.Model model = new Manifest.Model();
            model.setModelName(modelInfo.getModelName());
            model.setDescription(modelInfo.getDescription());
            model.setModelFormat(modelInfo.getFormat());
            model.setHandler(modelInfo.getService());
            model.setModelVersion("snapshot");
            model.setParametersFile(modelInfo.getParameters());
            model.setSymbolFile(modelInfo.getSymbol());
            manifest.setModel(model);
        }

        return manifest;
    }

    public static final class CreatedBy {

        @SerializedName("Author")
        private String author;

        @SerializedName("Author-Email")
        private String email;

        public CreatedBy() {}

        public String getAuthor() {
            return author;
        }

        public void setAuthor(String author) {
            this.author = author;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }
    }

    public static final class ModelInfo {

        @SerializedName("Parameters")
        private String parameters;

        @SerializedName("Symbol")
        private String symbol;

        @SerializedName("LegacySignature")
        private String signature;

        @SerializedName("Description")
        private String description;

        @SerializedName("Model-Format")
        private String format;

        @SerializedName("Model-Name")
        private String modelName;

        @SerializedName("Service")
        private String service;

        public ModelInfo() {}

        public String getParameters() {
            return parameters;
        }

        public void setParameters(String parameters) {
            this.parameters = parameters;
        }

        public String getSymbol() {
            return symbol;
        }

        public void setSymbol(String symbol) {
            this.symbol = symbol;
        }

        public String getSignature() {
            return signature;
        }

        public void setSignature(String signature) {
            this.signature = signature;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public String getFormat() {
            return format;
        }

        public void setFormat(String format) {
            this.format = format;
        }

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getService() {
            return service;
        }

        public void setService(String service) {
            this.service = service;
        }
    }
}
