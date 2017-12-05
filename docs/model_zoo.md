# Model Zoo

This page lists model archives that are pre-trained and pre-packaged, ready to be served for inference with MMS.
To propose a model for inclusion, please submit a [pull request](https://github.com/awslabs/mxnet-model-server/pulls).

*Special thanks to the [Apache MXNet](https://mxnet.incubator.apache.org) community whose Model Zoo and Model Examples were used in generating these model archives.*


| Model File | Type | Dataset | Size | Download |
| --- | --- | --- | --- | --- |
| [CaffeNet](https://s3.amazonaws.com/model-server/models/caffenet/caffenet.model) | Image Classification | ImageNet | 216 MB |
| [Inception v3 w/BatchNorm](#inception) | Image Classification | ImageNet | 45 MB | [.model](https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model) |
| [LSTM PTB](https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb.model) | Language Modeling | PennTreeBank | 16 MB |
| [Network in Network (NiN)](https://s3.amazonaws.com/model-server/models/nin/nin.model) | Image Classification | ImageNet | 30 MB |
| [ResNet-152](https://s3.amazonaws.com/model-server/models/resnet-152/resnet-152.model) | Image Classification | ImageNet | 241 MB |
| [ResNet-18](https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model) | Image Classification | ImageNet | 43 MB |
| [ResNet50-SSD](https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model) | SSD (Single Shot MultiBox Detector) | ImageNet | 124 MB |
| [ResNext101-64x4d](https://s3.amazonaws.com/model-server/models/resnext-101-64x4d/resnext-101-64x4d.model) | Image Classification | ImageNet | 334 MB |
| [SqueezeNet v1.1](https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model) | Image Classification | ImageNet | 5 MB |
| [VGG16](https://s3.amazonaws.com/model-server/models/vgg16/vgg16.model) | Image Classification | ImageNet | 490 MB |
| [VGG19](https://s3.amazonaws.com/model-server/models/vgg19/vgg19.model) | Image Classification | ImageNet | 509 MB |

## <a name="caffenet"></a>CaffeNet
Image classification trained on ImageNet.

"...a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. ".

[Krizhevsky, et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
### Start server
```bash
mxnet-model-server --models caffenet=https://s3.amazonaws.com/model-server/models/caffenet/caffenet.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/caffenet/predict -F "data=@kitten.jpeg"
```

## <a name="inception"></a>Inception v3
Image classification trained on ImageNet.

"...exploring ways to scale up networks in ways that aim at utilizing
the added computation as efficiently as possible by suitably
factorized convolutions and aggressive regularization."

[Szegedy, et al., Google](https://arxiv.org/pdf/1512.00567.pdf)
### Start server
```bash
mxnet-model-server --models inception-bn=https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/inception-bn/predict -F "data=@kitten.jpeg"
```
