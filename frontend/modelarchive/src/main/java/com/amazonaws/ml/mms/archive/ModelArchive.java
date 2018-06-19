package com.amazonaws.ml.mms.archive;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.Enumeration;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelArchive {

    private static final Logger logger = LoggerFactory.getLogger(ModelArchive.class);

    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static final Pattern URL_PATTERN =
            Pattern.compile("http(s)://.*", Pattern.CASE_INSENSITIVE);

    private static final String MANIFEST_FILE = "MANIFEST.json";
    private static final String SIGNATURE_FILE = "signature.json";

    private Manifest manifest;
    private Signature signature;
    private String url;
    private File modelDir;

    public ModelArchive(Manifest manifest, Signature signature, String url, File modelDir) {
        this.manifest = manifest;
        this.signature = signature;
        this.url = url;
        this.modelDir = modelDir;
    }

    public static ModelArchive downloadModel(String modelStore, String url)
            throws InvalidModelException {
        if (URL_PATTERN.matcher(url).matches()) {
            try {
                File modelDir = download(url);
                return load(url, modelDir, false);
            } catch (IOException e) {
                throw new InvalidModelException("Failed to download model archive: " + url, e);
            }
        }

        if (url.startsWith(".")) {
            throw new InvalidModelException("Invalid url: " + url);
        }

        File modelLocation = new File(modelStore, url);
        if (!modelLocation.exists()) {
            throw new InvalidModelException("Model not found: " + url);
        }
        if (url.endsWith(".model")) {
            try (InputStream is = new FileInputStream(modelLocation)) {
                modelLocation = unzip(is, null);
                return load(url, modelLocation, false);
            } catch (IOException e) {
                throw new InvalidModelException("Failed to unzip model archive: " + url, e);
            }
        }
        return load(url, modelLocation, true);
    }

    public static void migrate(File legacyModelFile, File destination)
            throws InvalidModelException {
        try (ZipFile zip = new ZipFile(legacyModelFile);
                ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(destination))) {

            ZipEntry manifestEntry = zip.getEntry(MANIFEST_FILE);
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
            ZipEntry signatureEntry = zip.getEntry(SIGNATURE_FILE);
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

            zos.putNextEntry(new ZipEntry(MANIFEST_FILE));
            zos.write(GSON.toJson(manifest).getBytes(StandardCharsets.UTF_8));

            zos.putNextEntry(new ZipEntry(SIGNATURE_FILE));
            zos.write(GSON.toJson(signature).getBytes(StandardCharsets.UTF_8));

            Enumeration<? extends ZipEntry> en = zip.entries();
            while (en.hasMoreElements()) {
                ZipEntry entry = en.nextElement();
                String name = entry.getName();
                if (SIGNATURE_FILE.equalsIgnoreCase(name)
                        || MANIFEST_FILE.equalsIgnoreCase(name)
                        || name.startsWith(".")) {
                    continue;
                }
                zos.putNextEntry(new ZipEntry(name));
                if (!entry.isDirectory()) {
                    IOUtils.copy(zip.getInputStream(entry), zos);
                }
            }
        } catch (IOException e) {
            FileUtils.deleteQuietly(destination);
            throw new InvalidModelException("Unable to extract model file.", e);
        }
    }

    private static File download(String path) throws IOException {
        URL url = new URL(path);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        if (conn.getResponseCode() != HttpURLConnection.HTTP_OK) {
            throw new IOException(
                    "Failed download model from: " + path + ", code: " + conn.getResponseCode());
        }

        String eTag = conn.getHeaderField("ETag");
        File tmpDir = new File(System.getProperty("java.io.tmpdir"));
        File modelDir = new File(tmpDir, "models");
        FileUtils.forceMkdir(modelDir);
        if (eTag != null) {
            if (eTag.startsWith("\"") && eTag.endsWith("\"") && eTag.length() > 2) {
                eTag = eTag.substring(1, eTag.length() - 1);
            }
            File dir = new File(modelDir, eTag);
            if (dir.exists()) {
                logger.info("model folder already exists: {}", eTag);
                return dir;
            }
        }
        return unzip(conn.getInputStream(), eTag);
    }

    private static ModelArchive load(String url, File dir, boolean copyOnMigrate)
            throws InvalidModelException {
        File manifestFile = findFile(dir, MANIFEST_FILE, true); // for 0.1 model archive
        if (manifestFile == null) {
            throw new InvalidModelException("MANIFEST.json file not found.");
        }
        File modelDir = manifestFile.getParentFile();
        File signatureFile = new File(modelDir, SIGNATURE_FILE);
        if (!signatureFile.exists()) {
            throw new InvalidModelException("signature.json file not found.");
        }

        try (Reader reader =
                new InputStreamReader(new FileInputStream(manifestFile), StandardCharsets.UTF_8)) {
            Manifest manifest;
            Signature signature;

            JsonParser parser = new JsonParser();
            JsonObject json = (JsonObject) parser.parse(reader);
            JsonPrimitive version = json.getAsJsonPrimitive("specificationVersion");

            if (version != null) {
                manifest = GSON.fromJson(json, Manifest.class);
                try (Reader r =
                        new InputStreamReader(
                                new FileInputStream(signatureFile), StandardCharsets.UTF_8)) {
                    signature = GSON.fromJson(r, Signature.class);
                } catch (IOException | JsonParseException e) {
                    throw new InvalidModelException("Failed to parse signature.json.", e);
                }
                ModelArchive archive = new ModelArchive(manifest, signature, url, modelDir);
                archive.validate();
                return archive;
            }

            // MMS 0.4
            LegacyManifest legacyManifest = GSON.fromJson(json, LegacyManifest.class);
            manifest = legacyManifest.migrate();

            try (Reader r =
                    new InputStreamReader(
                            new FileInputStream(signatureFile), StandardCharsets.UTF_8)) {
                LegacySignature legacySignature = GSON.fromJson(r, LegacySignature.class);
                signature = legacySignature.migrate();
            } catch (IOException | JsonParseException e) {
                throw new InvalidModelException("Failed to parse legacy signature.json.", e);
            }

            if (copyOnMigrate) {
                File tmpDir = FileUtils.getTempDirectory();
                File copyDir = new File(tmpDir, "models/" + manifest.getModel().getModelName());
                FileUtils.deleteDirectory(copyDir);
                FileUtils.forceMkdir(copyDir);
                FileUtils.copyDirectory(
                        dir,
                        copyDir,
                        f ->
                                !f.isHidden()
                                        && !SIGNATURE_FILE.equalsIgnoreCase(f.getName())
                                        && !MANIFEST_FILE.equalsIgnoreCase(f.getName()));
                modelDir = copyDir;
            }

            File output = new File(modelDir, MANIFEST_FILE);
            try (Writer writer =
                    new OutputStreamWriter(new FileOutputStream(output), StandardCharsets.UTF_8)) {
                writer.write(GSON.toJson(manifest));
            }

            output = new File(modelDir, SIGNATURE_FILE);
            try (Writer writer =
                    new OutputStreamWriter(new FileOutputStream(output), StandardCharsets.UTF_8)) {
                writer.write(GSON.toJson(signature));
            }

            ModelArchive archive = new ModelArchive(manifest, signature, url, modelDir);
            archive.validate();
            return archive;
        } catch (IOException | JsonParseException e) {
            throw new InvalidModelException("Failed to parse MANIFEST.json.", e);
        }
    }

    private static File findFile(File dir, String fileName, boolean recursive) {
        File[] list = dir.listFiles();
        if (list == null) {
            return null;
        }
        for (File file : list) {
            if (recursive && file.isDirectory()) {
                File f = findFile(file, fileName, false);
                if (f != null) {
                    return f;
                }
            } else if (file.getName().equalsIgnoreCase(fileName)) {
                return file;
            }
        }
        return null;
    }

    public static File unzip(InputStream is, String eTag) throws IOException {
        File tmpDir = FileUtils.getTempDirectory();
        File modelDir = new File(tmpDir, "models");
        FileUtils.forceMkdir(modelDir);

        File tmp = File.createTempFile("model", ".download");
        FileUtils.forceDelete(tmp);
        FileUtils.forceMkdir(tmp);

        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA1");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
        ZipUtils.unzip(new DigestInputStream(is, md), tmp);
        if (eTag == null) {
            eTag = Base64.getEncoder().encodeToString(md.digest());
        }
        File dir = new File(modelDir, eTag);
        if (dir.exists()) {
            FileUtils.deleteDirectory(tmp);
            logger.info("model folder already exists: {}", eTag);
            return dir;
        }

        FileUtils.moveDirectory(tmp, dir);

        return dir;
    }

    public void validate() throws InvalidModelException {
        Manifest.Model model = manifest.getModel();
        if (model == null) {
            throw new InvalidModelException("Missing Model entry in manifest file.");
        }

        if (model.getModelName() == null) {
            throw new InvalidModelException("Missing Model name in manifest file.");
        }

        if (model.getHandler() == null) {
            throw new InvalidModelException("Missing Model handler in manifest file.");
        }

        if (signature.getRequest() == null) {
            throw new InvalidModelException("Missing <Request> in signature.json.");
        }

        if (signature.getResponse() == null) {
            throw new InvalidModelException("Missing <Response> in signature.json.");
        }

        // TODO: Add more validation
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

    public File getModelDir() {
        return modelDir;
    }

    public String getModelName() {
        return manifest.getModel().getModelName();
    }

    public void clean() {
        if (url != null) {
            FileUtils.deleteQuietly(modelDir);
        }
    }
}
