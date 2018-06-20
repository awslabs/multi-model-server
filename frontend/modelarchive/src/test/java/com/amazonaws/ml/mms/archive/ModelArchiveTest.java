package com.amazonaws.ml.mms.archive;

import java.io.File;
import java.io.IOException;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ModelArchiveTest {

    private File output;

    @BeforeTest
    public void afterTest() {
        output = new File("build/tmp/test/noop.model");
        FileUtils.deleteQuietly(output);
        FileUtils.deleteQuietly(new File("build/tmp/test/noop"));
        File tmp = FileUtils.getTempDirectory();
        FileUtils.deleteQuietly(new File(tmp, "models"));
    }

    @Test
    public void test() throws InvalidModelException, IOException {
        String modelStore = "src/test/resources/models";

        // load 0.1 model archive
        ModelArchive archive = ModelArchive.downloadModel(modelStore, "noop-v0.1.model");
        Assert.assertEquals(archive.getModelName(), "noop_v0.1");

        // load 0.1 model from model directory
        File src = new File(modelStore, "noop-v0.1.model");
        File target = new File("build/tmp/test/noop");
        FileUtils.forceMkdir(target);
        ZipUtils.unzip(src, target);
        archive = ModelArchive.downloadModel("build/tmp/test", "noop");
        Assert.assertEquals(archive.getModelName(), "noop_v0.1");

        // load model for s3
        archive =
                ModelArchive.downloadModel(
                        modelStore,
                        "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        Assert.assertEquals(archive.getModelName(), "squeezenet_v1.1");

        String[] args = new String[4];
        args[0] = "export";
        args[1] = "--model-name=noop";
        args[2] = "--model-path=" + archive.getModelDir();
        args[3] = "--output-file=" + output.getAbsolutePath();

        Exporter.main(args);
        Assert.assertTrue(output.exists());

        FileUtils.forceDelete(output);

        File modelFile = new File(modelStore, "noop-v0.1.model");
        ModelArchive.migrate(modelFile, output);
        Assert.assertTrue(output.exists());

        // load 1.0 model
        archive =
                ModelArchive.downloadModel(
                        output.getParentFile().getAbsolutePath(), output.getName());
        Assert.assertEquals(archive.getModelName(), "noop_v0.1");
    }
}
