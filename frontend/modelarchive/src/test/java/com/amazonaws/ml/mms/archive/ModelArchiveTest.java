/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
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
        FileUtils.deleteQuietly(new File("build/tmp/test/noop-v0.1.model"));
        File tmp = FileUtils.getTempDirectory();
        FileUtils.deleteQuietly(new File(tmp, "models"));
    }

    @Test
    public void test() throws InvalidModelException, IOException {
        String modelStore = "src/test/resources/models";

        // load 0.1 model from model folder
        ModelArchive archive = ModelArchive.downloadModel(modelStore, "noop-v0.1");
        Assert.assertEquals(archive.getModelName(), "noop_v0.1");

        // load 0.1 model from model archive
        File src = new File(modelStore, "noop-v0.1");
        File target = new File("build/tmp/test", "noop-v0.1.model");
        ZipUtils.zip(src, target, false);
        archive = ModelArchive.downloadModel("build/tmp/test", "noop-v0.1.model");
        Assert.assertEquals(archive.getModelName(), "noop_v0.1");

        // load model for s3
        archive =
                ModelArchive.downloadModel(
                        modelStore,
                        "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        Assert.assertEquals(archive.getModelName(), "squeezenet_v1.1");

        // test export
        String[] args = new String[4];
        args[0] = "export";
        args[1] = "--model-name=noop";
        args[2] = "--model-path=" + archive.getModelDir();
        args[3] = "--output-file=" + output.getAbsolutePath();

        Exporter.main(args);
        Assert.assertTrue(output.exists());

        FileUtils.forceDelete(output);

        ModelArchive.migrate(target, output);
        Assert.assertTrue(output.exists());

        // load 1.0 model
        archive = ModelArchive.downloadModel(modelStore, "noop-v1.0");
        Assert.assertEquals(archive.getModelName(), "noop");
    }
}
