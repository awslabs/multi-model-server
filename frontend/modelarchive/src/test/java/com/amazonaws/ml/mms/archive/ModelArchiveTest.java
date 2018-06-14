package com.amazonaws.ml.mms.archive;

import java.io.File;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ModelArchiveTest {

    @BeforeTest
    public void afterTest() throws IOException {
        File output = new File("build/tmp/test/noop.model");
        if (output.exists() && !output.delete()) {
            throw new IOException("Unable to delete file: " + output);
        }
    }

    @Test
    public void test() throws InvalidModelException, IOException {
        File file = new File("src/test/resources/models/noop-v0.1.model");
        ModelArchive archive = ModelArchive.parseModelMetadata(file);
        Assert.assertEquals(archive.getModelName(), "noop_v0.1");

        File modelPath = new File("build/tmp/test/noop");
        if (!modelPath.exists() && !modelPath.mkdirs()) {
            throw new IOException("Unable to create dir: " + modelPath);
        }
        ZipUtils.unzip(file, modelPath);

        File output = new File("build/tmp/test/noop.model");

        String[] args = new String[3];
        args[0] = "export";
        args[1] = "--model-name=noop";
        args[2] = "--model-path=build/tmp/test/noop";
        Exporter.main(args);
        Assert.assertTrue(output.exists());

        if (!output.delete()) {
            throw new IOException("Unable to delete file: " + output);
        }

        ModelArchive.migrate(file, output);
        Assert.assertTrue(output.exists());

        archive = ModelArchive.parseModelMetadata(output);
        Assert.assertEquals(archive.getModelName(), "noop_v0.1");
    }
}
