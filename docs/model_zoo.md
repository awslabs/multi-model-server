# Model Zoo

This page lists model archives that are pre-trained and pre-packaged, ready to be served for inference with MMS.
To propose a model for inclusion, please submit a [pull request](https://github.com/awslabs/mxnet-model-server/pulls).

*Special thanks to the [Apache MXNet](https://mxnet.incubator.apache.org) community whose Model Zoo and Model Examples were used in generating these model archives.*


| Model File | Type | Dataset | Source | Size | Download |
| --- | --- | --- | --- | --- | --- |
| [AlexNet](#alexnet) | Image Classification | ImageNet | ONNX | 233 MB | [.model](https://s3.amazonaws.com/model-server/models/onnx-alexnet/alexnet.model) |
| [ArcFace-ResNet100](#arcface-resnet100_onnx) | Face Recognition | Refined MS-Celeb1M | ONNX | 236.4 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/arcface-resnet100.model) |
| [CaffeNet](#caffenet) | Image Classification | ImageNet | MXNet | 216 MB | [.model](https://s3.amazonaws.com/model-server/models/caffenet/caffenet.model) |
| [FERPlus](#ferplus_onnx) | Emotion Detection | FER2013 | ONNX | 35MB | [.model](https://s3.amazonaws.com/model-server/models/FERPlus/FERPlus.model) |
| [Inception v1](#inception_v1) | Image Classification | ImageNet | ONNX | 27 MB | [.model](https://s3.amazonaws.com/model-server/models/onnx-inception_v1/inception_v1.model) |
| [Inception v3 w/BatchNorm](#inception_v3) | Image Classification | ImageNet | MXNet | 45 MB |  [.model](https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model) |
| [LSTM PTB](#lstm-ptb) | Language Modeling | PennTreeBank | MXNet | 16 MB | [.model](https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb.model) |
| [MobileNetv2-1.0](#mobilenetv2-1.0_onnx) | Image Classification | ImageNet | ONNX | 13.7 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-mobilenet/mobilenetv2-1.0.model) |
| [Network in Network (NiN)](#nin) | Image Classification | ImageNet | MXNet | 30 MB | [.model](https://s3.amazonaws.com/model-server/models/nin/nin.model) |
| [ResNet-152](#resnet-152) | Image Classification | ImageNet | MXNet | 241 MB | [.model](https://s3.amazonaws.com/model-server/models/resnet-152/resnet-152.model) |
| [ResNet-18](#resnet-18) | Image Classification | ImageNet | MXNet | 43 MB | [.model](https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model) |
| [ResNet50-SSD](#resnet50-ssd) | SSD (Single Shot MultiBox Detector) | ImageNet | MXNet | 124 MB | [.model](https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model) |
| [ResNext101-64x4d](#resnext101) | Image Classification | ImageNet | MXNet | 334 MB | [.model](https://s3.amazonaws.com/model-server/models/resnext-101-64x4d/resnext-101-64x4d.model) |
| [ResNet-18v1](#resnet-18v1) | Image Classification | ImageNet | ONNX | 45 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet18v1.model) |
| [ResNet-34v1](#resnet-34v1) | Image Classification | ImageNet | ONNX | 83 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet34v1.model) |
| [ResNet-50v1](#resnet-50v1) | Image Classification | ImageNet | ONNX | 98 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet50v1.model) |
| [ResNet-101v1](#resnet-101v1) | Image Classification | ImageNet | ONNX | 171 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet101v1.model) |
| [ResNet-152v1](#resnet-152v1) | Image Classification | ImageNet | ONNX | 231 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet152v1.model) |
| [ResNet-18v2](#resnet-18v2) | Image Classification | ImageNet | ONNX | 45 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet18v2.model) |
| [ResNet-34v2](#resnet-34v2) | Image Classification | ImageNet | ONNX | 83 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet34v2.model) |
| [ResNet-50v2](#resnet-50v2) | Image Classification | ImageNet | ONNX | 98 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet50v2.model) |
| [ResNet-101v2](#resnet-101v2) | Image Classification | ImageNet | ONNX | 171 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet101v2.model) |
| [ResNet-152v2](#resnet-152v2) | Image Classification | ImageNet | ONNX | 231 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet152v2.model) |
| [SqueezeNet](#squeezenet) | Image Classification | ImageNet | ONNX | 5 MB | [.model](https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.model) |
| [SqueezeNet v1.1](#squeezenet_v1.1) | Image Classification | ImageNet | MXNet | 5 MB | [.model](https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model) |
| [SqueezeNet v1.1](#squeezenet_v1.1_onnx) | Image Classification | ImageNet | ONNX | 5 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/squeezenet_v1.1.model) |
| [VGG16](#vgg16) | Image Classification | ImageNet | MXNet | 490 MB | [.model](https://s3.amazonaws.com/model-server/models/vgg16/vgg16.model) |
| [VGG16](#vgg16_onnx) | Image Classification | ImageNet | ONNX | 527 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-vgg16/vgg16.model) |
| [VGG16_bn](#vgg16_bn_onnx) | Image Classification | ImageNet | ONNX | 527 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-vgg16_bn/vgg16_bn.model) |
| [VGG19](#vgg19) | Image Classification | ImageNet | MXNet | 509 MB | [.model](https://s3.amazonaws.com/model-server/models/vgg19/vgg19.model) |
| [VGG19](#vgg19_onnx) | Image Classification | ImageNet | ONNX | 548 MB | [.model](https://s3.amazonaws.com/model-server/models/onnx-vgg19/vgg19.model) |
| [VGG19_bn](#vgg19_bn_onnx) | Image Classification | ImageNet | ONNX | 548 MB | [.model](https://s3.amazonaws.com/mxnet-model-server/onnx-vgg19_bn/vgg19_bn.model) |


## Details on Each Model
Each model below comes with a basic description, and where available, a link to a scholarly article about the model.

Many of these models use a kitten image to test inference. Use the following to get one that will work:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
```


## <a name="alexnet"></a>AlexNet
* **Type**: Image classification trained on ImageNet

* **Reference**: [Krizhevsky, et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models alexnet=https://s3.amazonaws.com/model-server/models/onnx-alexnet/alexnet.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/alexnet/predict -F "input_0=@kitten.jpeg"
```

## <a name="arcface-resnet100_onnx"></a>ArcFace-ResNet100 (from ONNX model zoo)
* **Type**: Face Recognition model trained on refined MS-Celeb1M dataset (model imported from ONNX)

* **Reference**: [Deng et al.](https://arxiv.org/abs/1801.07698)

* **Model Service**: [arcface_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/arcface_service.py)

* **Install dependencies**:
```bash
pip install opencv-python
pip install scikit-learn
pip install easydict
pip install scikit-image
pip install numpy
```

* **Start Server**:
```bash
mxnet-model-server --models arcface=https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/arcface-resnet100.model
```

* **Get two test images**:
```bash
curl -O https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/input1.jpg

curl -O https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/input2.jpg
```

* **Download inference script**:

The script makes two inference calls to the server for the two input images and computes the similarity scores using output embeddings.

```bash
curl -O https://s3.amazonaws.com/mxnet-model-server/onnx-arcface/arcface_inference.sh
```

* **Run Prediction**:
```bash
bash arcface_inference.sh arcface input1.jpg input2.jpg
```


## <a name="caffenet"></a>CaffeNet
* **Type**: Image classification trained on ImageNet

* **Reference**: [Krizhevsky, et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models caffenet=https://s3.amazonaws.com/model-server/models/caffenet/caffenet.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/caffenet/predict -F "data=@kitten.jpeg"
```

## <a name="ferplus_onnx"></a>FERPlus
* **Type**: Emotion detection trained on FER2013 dataset (model imported from ONNX)

* **Reference**: [Barsoum et al.](https://arxiv.org/abs/1608.01041)

* **Model Service**: [emotion_detection_service.py](https://s3.amazonaws.com/model-server/models/FERPlus/emotion_detection_service.py)

* **Start Server**:
```bash
mxnet-model-server --models emotion_detection=https://s3.amazonaws.com/model-server/models/FERPlus/FERPlus.model
```

* **Get a test image**:
```bash
curl -O https://s3.amazonaws.com/model-server/models/FERPlus/input.jpg
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/emotion_detection/predict -F "Input2505=@input.jpg"
```


## <a name="inception_v1"></a>Inception v1
* **Type**: Image classification trained on ImageNet

* **Reference**: [Szegedy, et al., Google](https://arxiv.org/pdf/1512.00567.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models inception-v1=https://s3.amazonaws.com/model-server/models/onnx-inception_v1/inception_v1.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/inception-v1/predict -F "input_0=@kitten.jpeg"
```


## <a name="inception_v3"></a>Inception v3
* **Type**: Image classification trained on ImageNet

* **Reference**: [Szegedy, et al., Google](https://arxiv.org/pdf/1512.00567.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models inception-bn=https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/inception-bn/predict -F "data=@kitten.jpeg"
```


## <a name="lstm-ptb"></a>LSTM PTB
Long short-term memory network trained on the PennTreeBank dataset.

* **Reference**: [Hochreiter, et al.](http://www.bioinf.jku.at/publications/older/2604.pdf)

* **Model Service**: [lstm_ptb_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/lstm_ptb/lstm_ptb_service.py)

* **Start Server**:
```bash
mxnet-model-server --models lstm_ptb=https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/lstm_ptb/predict -F "data=[{'input_sentence': 'on the exchange floor as soon as ual stopped trading we <unk> for a panic said one top floor trader'}]"
```

## <a name="mobilenetv2-1.0_onnx"></a>MobileNetv2-1.0 (from ONNX model zoo)
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Sandler et al.](https://arxiv.org/abs/1801.04381)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models mobilenet=https://s3.amazonaws.com/mxnet-model-server/onnx-mobilenet/mobilenetv2-1.0.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/mobilenet/predict -F "input_0=@kitten.jpeg"
```


## <a name="nin"></a>Network in Network
* **Type**: Image classification trained on ImageNet

* **Reference**: [Lin, et al.](https://arxiv.org/pdf/1312.4400v3.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models nin=https://s3.amazonaws.com/model-server/models/nin/nin.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/nin/predict -F "data=@kitten.jpeg"
```


## <a name="resnet-152"></a>ResNet-152
* **Type**: Image classification trained on ImageNet

* **Reference**: [Lin, et al.](https://arxiv.org/pdf/1312.4400v3.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet-152=https://s3.amazonaws.com/model-server/models/resnet-152/resnet-152.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/resnet-152/predict -F "data=@kitten.jpeg"
```


## <a name="resnet-18"></a>ResNet-18
* **Type**: Image classification trained on ImageNet

* **Reference**: [He, et al.](https://arxiv.org/pdf/1512.03385v1.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet-18=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "data=@kitten.jpeg"
```


## <a name="resnet50-ssd"></a>ResNet50-SSD
* **Type**: Image classification trained on ImageNet

* **Reference**: [Liu, et al.](https://arxiv.org/pdf/1512.02325v4.pdf)

* **Model Service**: [ssd_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/ssd_service.py)

* **Start Server**:
```bash
mxnet-model-server --models SSD=https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model
```

* **Run Prediction**:
```bash
wget https://www.dphotographer.co.uk/users/21963/thm1024/1337890426_Img_8133.jpg
curl -X POST http://127.0.0.1:8080/SSD/predict -F "data=@1337890426_Img_8133.jpg"
```


## <a name="resnext101"></a>ResNext101-64x4d
* **Type**: Image classification trained on ImageNet

* **Reference**: [Xie, et al.](https://arxiv.org/pdf/1611.05431.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnext101=https://s3.amazonaws.com/model-server/models/resnext-101-64x4d/resnext-101-64x4d.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/resnext101/predict -F "data=@kitten.jpeg"
```

## <a name="resnet_header"></a>ResNet (from ONNX model zoo)

### <a name="resnet-18v1"></a>ResNet18-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet18-v1=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet18v1.model
```

### <a name="resnet-34v1"></a>ResNet34-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet34-v1=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet34v1.model
```

### <a name="resnet-50v1"></a>ResNet50-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet50-v1=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet50v1.model
```

### <a name="resnet-101v1"></a>ResNet101-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet101-v1=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet101v1.model
```

### <a name="resnet-152v1"></a>ResNet152-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet152-v1=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv1/resnet152v1.model
```

### <a name="resnet-18v2"></a>ResNet18-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet18-v2=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet18v2.model
```

### <a name="resnet-34v2"></a>ResNet34-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet34-v2=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet34v2.model
```

### <a name="resnet-50v2"></a>ResNet50-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet50-v2=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet50v2.model
```

### <a name="resnet-101v2"></a>ResNet101-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet101-v2=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet101v2.model
```

### <a name="resnet-152v2"></a>ResNet152-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models resnet152-v2=https://s3.amazonaws.com/mxnet-model-server/onnx-resnetv2/resnet152v2.model
```


## <a name="squeezenet"></a>SqueezeNet
* **Type**: Image classification trained on ImageNet

* **Reference**: [Iandola, et al.](https://arxiv.org/pdf/1602.07360v4.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models squeezenet=https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "input_0=@kitten.jpeg"
```


## <a name="squeezenet_v1.1"></a>SqueezeNet v1.1
* **Type**: Image classification trained on ImageNet

* **Reference**: [Iandola, et al.](https://arxiv.org/pdf/1602.07360v4.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models squeezenet_v1.1=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/squeezenet_v1.1/predict -F "data=@kitten.jpeg"
```

## <a name="squeezenet_v1.1_onnx"></a>SqueezeNet v1.1 (from ONNX model zoo)
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Iandola, et al.](https://arxiv.org/pdf/1602.07360v4.pdf)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models squeezenet_v1.1=https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/squeezenet_v1.1.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/squeezenet_v1.1/predict -F "data=@kitten.jpeg"
```


## <a name="vgg16"></a>VGG16
* **Type**: Image classification trained on ImageNet

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models vgg16=https://s3.amazonaws.com/model-server/models/vgg16/vgg16.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/vgg16/predict -F "data=@kitten.jpeg"
```

## <a name="vgg19"></a>VGG19
* **Type**: Image classification trained on ImageNet

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models vgg19=https://s3.amazonaws.com/model-server/models/vgg19/vgg19.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/vgg19/predict -F "data=@kitten.jpeg"
```

## <a name="vgg_header"></a>VGG (from ONNX model zoo)
### <a name="vgg16_onnx"></a>VGG16
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models vgg16=https://s3.amazonaws.com/mxnet-model-server/onnx-vgg16/vgg16.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/vgg16/predict -F "data=@kitten.jpeg"
```

### <a name="vgg16_bn_onnx"></a>VGG16_bn
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf) (Batch normalization applied after each conv layer of VGG16)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models vgg16_bn=https://s3.amazonaws.com/mxnet-model-server/onnx-vgg16_bn/vgg16_bn.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/vgg16_bn/predict -F "data=@kitten.jpeg"
```

### <a name="vgg19_onnx"></a>VGG19
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/models/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models vgg19=https://s3.amazonaws.com/model-server/models/onnx-vgg19/vgg19.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/vgg19/predict -F "input_0=@kitten.jpeg"
```

### <a name="vgg19_bn_onnx"></a>VGG19_bn
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf) (Batch normalization applied after each conv layer of VGG19)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/mxnet-model-server/onnx-squeezenet_v1.1/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --models vgg19=https://s3.amazonaws.com/mxnet-model-server/onnx-vgg19_bn/vgg19_bn.model
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/vgg19_bn/predict -F "data=@kitten.jpeg"
```