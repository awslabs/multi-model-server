package com.amazonaws.ml.mms.archive;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

public class ModelArchive {

    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private Manifest manifest;
    private Signature signature;
    private String url;

    public ModelArchive(Manifest manifest, Signature signature, String url) {
        this.manifest = manifest;
        this.signature = signature;
        this.url = url;
    }

    public static ModelArchive parseModelMetadata(File modelFile) throws InvalidModelException {
        try (ZipFile zip = new ZipFile(modelFile)) {
            ZipEntry manifestEntry = zip.getEntry("MANIFEST.json");
            if (manifestEntry == null) {
                throw new InvalidModelException("Missing manifest file in model archive.");
            }
            ZipEntry signatureEntry = zip.getEntry("signature.json");
            if (signatureEntry == null) {
                throw new InvalidModelException("Missing signature file in model archive.");
            }

            InputStream is = zip.getInputStream(manifestEntry);
            Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8);
            JsonParser parser = new JsonParser();
            JsonObject json = (JsonObject) parser.parse(reader);
            JsonPrimitive version = json.getAsJsonPrimitive("specificationVersion");

            is = zip.getInputStream(signatureEntry);
            reader = new InputStreamReader(is, StandardCharsets.UTF_8);

            Manifest manifest;
            Signature signature;
            if (version == null) {
                // MMS 0.4
                LegacyManifest legacyManifest = GSON.fromJson(json, LegacyManifest.class);
                manifest = legacyManifest.migrate();

                LegacySignature legacySignature = GSON.fromJson(reader, LegacySignature.class);
                signature = legacySignature.migrate();
            } else {
                manifest = GSON.fromJson(json, Manifest.class);
                signature = GSON.fromJson(reader, Signature.class);
            }

            if (manifest.getModel() == null) {
                throw new InvalidModelException("Missing Model entry in manifest file.");
            }

            return new ModelArchive(manifest, signature, modelFile.getAbsolutePath());
        } catch (IOException e) {
            throw new InvalidModelException("Unable to extract model file.", e);
        }
    }

    public static void migrate(File legacyModelFile, File destination)
            throws InvalidModelException {

        try (ZipFile zip = new ZipFile(legacyModelFile);
                ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(destination))) {

            ZipEntry manifestEntry = zip.getEntry("MANIFEST.json");
            if (manifestEntry == null) {
                throw new InvalidModelException("Missing manifest file in model archive.");
            }

            InputStream is = zip.getInputStream(manifestEntry);
            Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8);

            JsonParser parser = new JsonParser();
            JsonObject json = (JsonObject) parser.parse(reader);
            JsonPrimitive version = json.getAsJsonPrimitive("specificationVersion");
            Manifest manifest;
            if (version != null && "1.0".equals(version.getAsString())) {
                throw new IllegalArgumentException("model archive is already in 1.0 version.");
            }

            LegacyManifest legacyManifest = GSON.fromJson(json, LegacyManifest.class);
            ZipEntry signatureEntry = zip.getEntry("signature.json");
            if (signatureEntry == null) {
                throw new InvalidModelException("Missing signature file in model archive.");
            }
            is = zip.getInputStream(signatureEntry);
            reader = new InputStreamReader(is, StandardCharsets.UTF_8);
            LegacySignature legacySignature = GSON.fromJson(reader, LegacySignature.class);
            Signature signature = legacySignature.migrate();
            manifest = legacyManifest.migrate();
            if (manifest.getModel() == null) {
                throw new InvalidModelException("Missing Model entry in manifest file.");
            }

            zos.putNextEntry(new ZipEntry("MANIFEST.json"));
            zos.write(GSON.toJson(manifest).getBytes(StandardCharsets.UTF_8));

            zos.putNextEntry(new ZipEntry("signature.json"));
            zos.write(GSON.toJson(signature).getBytes(StandardCharsets.UTF_8));

            Enumeration<? extends ZipEntry> en = zip.entries();
            while (en.hasMoreElements()) {
                ZipEntry entry = en.nextElement();
                String name = entry.getName();
                if ("signature.json".equalsIgnoreCase(name)
                        || "MANIFEST.json".equalsIgnoreCase(name)
                        || name.startsWith(".")) {
                    continue;
                }
                zos.putNextEntry(new ZipEntry(name));
                if (!entry.isDirectory()) {
                    ZipUtils.copy(zip.getInputStream(entry), zos);
                }
            }
        } catch (IOException e) {
            if (!destination.delete()) {
                destination.deleteOnExit();
            }
            throw new InvalidModelException("Unable to extract model file.", e);
        }
    }

    public Manifest getManifest() {
        return manifest;
    }

    public Signature getSignature() {
        return signature;
    }

    public String getUrl() {
        return url;
    }

    public String getModelName() {
        return manifest.getModel().getModelName();
    }
}
