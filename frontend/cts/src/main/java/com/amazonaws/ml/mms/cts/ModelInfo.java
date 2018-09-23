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
package com.amazonaws.ml.mms.cts;

public class ModelInfo {

    public static final int IMAGE_CLASSIFICATION = 1;
    public static final int FACE_RECOGNITION = 2;
    public static final int SEMANTIC_SEGMENTATION = 3;
    public static final int EMOTION_DETECTION = 4;
    public static final int LANGUAGE_MODELING = 5;

    private static final String S3_PREFIX =
            "https://s3.amazonaws.com/model-server/model_archive_1.0/";
    private static final String S3_PREFIX_LEGACY = "https://s3.amazonaws.com/model-server/models/";

    static final ModelInfo[] MODEL_ARCHIVE_1 = {
        new ModelInfo("FERPlus", ModelInfo.EMOTION_DETECTION),
        new ModelInfo("caffenet"),
        new ModelInfo("inception-bn"),
        new ModelInfo("lstm_ptb", ModelInfo.LANGUAGE_MODELING),
        new ModelInfo("nin"),
        new ModelInfo("onnx-arcface-resnet100", ModelInfo.FACE_RECOGNITION),
        new ModelInfo("onnx-duc", ModelInfo.SEMANTIC_SEGMENTATION),
        new ModelInfo("onnx-inception_v1"),
        new ModelInfo("onnx-mobilenet"),
        new ModelInfo("onnx-resnet101v1"),
        new ModelInfo("onnx-resnet101v2"),
        new ModelInfo("onnx-resnet152v1"),
        new ModelInfo("onnx-resnet152v2"),
        new ModelInfo("onnx-resnet18v1"),
        new ModelInfo("onnx-resnet18v2"),
        new ModelInfo("onnx-resnet34v1"),
        new ModelInfo("onnx-resnet34v2"),
        new ModelInfo("onnx-resnet50v1"),
        new ModelInfo("onnx-resnet50v2"),
        new ModelInfo("onnx-squeezenet"),
        new ModelInfo("onnx-vgg16"),
        new ModelInfo("onnx-vgg16_bn"),
        new ModelInfo("onnx-vgg19"),
        new ModelInfo("onnx-vgg19_bn"),
        new ModelInfo("resnet-152"),
        new ModelInfo("resnet-18"),
        new ModelInfo("resnet50_ssd"),
        new ModelInfo("resnext-101-64x4d"),
        new ModelInfo("squeezenet_v1.1"),
        new ModelInfo("squeezenet_v1.2"),
        new ModelInfo("vgg16"),
        new ModelInfo("vgg19")
    };

    static final ModelInfo[] MODEL_ARCHIVE_04 = {
        new ModelInfo(
                "FERPlus",
                "https://s3.amazonaws.com/model-server/models/FERPlus/ferplus.model",
                EMOTION_DETECTION),
        new ModelInfo(true, "caffenet"),
        new ModelInfo(
                "inception-bn",
                "https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model"),
        new ModelInfo(true, "lstm_ptb", LANGUAGE_MODELING),
        new ModelInfo(true, "nin"),
        new ModelInfo(
                "onnx-arcface-resnet100",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/arcface-resnet100.model"),
        new ModelInfo(
                "onnx-duc",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-duc/ResNet_DUC_HDC.model"),
        new ModelInfo(
                "onnx-inception_v1",
                "https://s3.amazonaws.com/model-server/models/onnx-inception_v1/inception_v1.model"),
        new ModelInfo(
                "onnx-mobilenet",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-mobilenet/mobilenetv2-1.0.model"),
        new ModelInfo(
                "onnx-resnet101v1",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet101v1.model"),
        new ModelInfo(
                "onnx-resnet101v2",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet101v2.model"),
        new ModelInfo(
                "onnx-resnet152v1",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet152v1.model"),
        new ModelInfo(
                "onnx-resnet152v2",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet152v2.model"),
        new ModelInfo(
                "onnx-resnet18v1",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet18v1.model"),
        new ModelInfo(
                "onnx-resnet18v2",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet18v2.model"),
        new ModelInfo(
                "onnx-resnet34v1",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet34v1.model"),
        new ModelInfo(
                "onnx-resnet34v2",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet34v2.model"),
        new ModelInfo(
                "onnx-resnet50v1",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet50v1.model"),
        new ModelInfo(
                "onnx-resnet50v2",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet50v2.model"),
        new ModelInfo(
                "onnx-squeezenet",
                "https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.model"),
        new ModelInfo(
                "onnx-vgg16", "https://s3.amazonaws.com/mxnet-model-server/onnx-vgg16/vgg16.model"),
        new ModelInfo(
                "onnx-vgg16_bn",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-vgg16_bn/vgg16_bn.model"),
        new ModelInfo(
                "onnx-vgg19",
                "https://s3.amazonaws.com/model-server/models/onnx-vgg19/vgg19.model"),
        new ModelInfo(
                "onnx-vgg19_bn",
                "https://s3.amazonaws.com/mxnet-model-server/onnx-vgg19_bn/vgg19_bn.model"),
        new ModelInfo(true, "resnet-152"),
        new ModelInfo(true, "resnet-18"),
        new ModelInfo(
                "resnet50_ssd",
                "https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model"),
        new ModelInfo(true, "resnext-101-64x4d"),
        new ModelInfo(true, "squeezenet_v1.1"),
        new ModelInfo(true, "vgg16"),
        new ModelInfo(true, "vgg19")
    };

    private String modelName;
    private String url;
    private int type;

    public ModelInfo(String modelName) {
        this(false, modelName, IMAGE_CLASSIFICATION);
    }

    public ModelInfo(String modelName, int type) {
        this(false, modelName, type);
    }

    public ModelInfo(boolean legacy, String modelName) {
        this(legacy, modelName, IMAGE_CLASSIFICATION);
    }

    public ModelInfo(boolean legacy, String modelName, int type) {
        this.modelName = modelName;
        if (legacy) {
            url = S3_PREFIX_LEGACY + modelName + '/' + modelName + ".model";
        } else {
            url = S3_PREFIX + modelName + ".mar";
        }
        this.type = type;
    }

    public ModelInfo(String modelName, String url) {
        this(modelName, url, IMAGE_CLASSIFICATION);
    }

    public ModelInfo(String modelName, String url, int type) {
        this.modelName = modelName;
        this.url = url;
        this.type = type;
    }

    public String getModelName() {
        return modelName;
    }

    public String getUrl() {
        return url;
    }

    public int getType() {
        return type;
    }
}
