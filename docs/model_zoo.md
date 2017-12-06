# Model Zoo

This page lists model archives that are pre-trained and pre-packaged, ready to be served for inference with MMS.
To propose a model for inclusion, please submit a [pull request](https://github.com/awslabs/mxnet-model-server/pulls).

*Special thanks to the [Apache MXNet](https://mxnet.incubator.apache.org) community whose Model Zoo and Model Examples were used in generating these model archives.*


| Model File | Type | Dataset | Size | Download |
| --- | --- | --- | --- | --- |
| [CaffeNet](#caffenet) | Image Classification | ImageNet | 216 MB | [.model](https://s3.amazonaws.com/model-server/models/caffenet/caffenet.model) |
| [Inception v3 w/BatchNorm](#inception) | Image Classification | ImageNet | 45 MB |  [.model](https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model) |
| [LSTM PTB](#lstm-ptb) | Language Modeling | PennTreeBank | 16 MB | [.model](https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb.model) |
| [Network in Network (NiN)](#nin) | Image Classification | ImageNet | 30 MB | [.model](https://s3.amazonaws.com/model-server/models/nin/nin.model) |
| [ResNet-152](#resnet-152) | Image Classification | ImageNet | 241 MB | [.model](https://s3.amazonaws.com/model-server/models/resnet-152/resnet-152.model) |
| [ResNet-18](#resnet-18) | Image Classification | ImageNet | 43 MB | [.model](https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model) |
| [ResNet50-SSD](#resnet50-ssd) | SSD (Single Shot MultiBox Detector) | ImageNet | 124 MB | [.model](https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model) |
| [ResNext101-64x4d](#resnext101) | Image Classification | ImageNet | 334 MB | [.model](https://s3.amazonaws.com/model-server/models/resnext-101-64x4d/resnext-101-64x4d.model) |
| [SqueezeNet v1.1](#squeezenet) | Image Classification | ImageNet | 5 MB | [.model](https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model) |
| [VGG16](#vgg16) | Image Classification | ImageNet | 490 MB | [.model](https://s3.amazonaws.com/model-server/models/vgg16/vgg16.model) |
| [VGG19](#vgg19) | Image Classification | ImageNet | 509 MB | [.model](https://s3.amazonaws.com/model-server/models/vgg19/vgg19.model) |

## Details on Each Model
Each model below comes with a basic description, a link to a scholarly article about the model, and extract from the article's abstract.

Many of these models use a kitten image to test inference. Use the following to get one that will work:
```bash
wget -O kitten.jpg \
  https://upload.wikimedia.org/wikipedia/commons/8/8f/Cute-kittens-12929201-1600-1200.jpg
```

## <a name="caffenet"></a>CaffeNet
Image classification trained on ImageNet.

[Krizhevsky, et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) "...a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. ".



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

[Szegedy, et al., Google](https://arxiv.org/pdf/1512.00567.pdf) "...exploring ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization."

### Start server
```bash
mxnet-model-server --models inception-bn=https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/inception-bn/predict -F "data=@kitten.jpeg"
```


## <a name="lstm-ptb"></a>LSTM PTB
Long short-term memory network trained on the PennTreeBank dataset.

[Hochreiter, et al.](http://www.bioinf.jku.at/publications/older/2604.pdf) "...introducing a novel, efficient, gradient-based method called Long Short-Term Memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing constant error flow through constant error carrousels within special units.

### Start server
```bash
mxnet-model-server --models lstm_ptb=https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/lstm_ptb/predict -F "data=[{'input_sentence': 'on the exchange floor as soon as ual stopped trading we <unk> for a panic said one top floor trader'}]"
```


## <a name="nin"></a>Network in Network
Image classification trained on ImageNet.

[Lin, et al.](https://arxiv.org/pdf/1312.4400v3.pdf) "...a novel deep network structure called “Network In Network”(NIN) to enhance model discriminability for local patches within the receptive field. The conventional convolutional layer uses linear filters followed by a nonlinear activation function to scan the input. Instead, we build micro neural networks with
more complex structures to abstract the data within the receptive field. We instantiate the micro neural network with a multilayer perceptron, which is a potent function approximator."

### Start server
```bash
mxnet-model-server --models nin=https://s3.amazonaws.com/model-server/models/nin/nin.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/nin/predict -F "data=@kitten.jpeg"
```


## <a name="resnet-152"></a>ResNet-152
Image classification trained on ImageNet.

[Lin, et al.](https://arxiv.org/pdf/1312.4400v3.pdf) "...a novel deep network structure called “Network In Network”(NIN) to enhance model discriminability for local patches within the receptive field. The conventional convolutional layer uses linear filters followed by a nonlinear activation function to scan the input. Instead, we build micro neural networks with more complex structures to abstract the data within the receptive field. We instantiate the micro neural network with a multilayer perceptron, which is a potent function approximator."

### Start server
```bash
mxnet-model-server --models resnet-152=https://s3.amazonaws.com/model-server/models/resnet-152/resnet-152.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/resnet-152/predict -F "data=@kitten.jpeg"
```


## <a name="resnet-18"></a>ResNet-18
Image classification trained on ImageNet.

[He, et al.](https://arxiv.org/pdf/1512.03385v1.pdf) "...a residual learning framework to ease the training of networks that are substantially deeper than those used
previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. "

### Start server
```bash
mxnet-model-server --models resnet-18=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "data=@kitten.jpeg"
```


## <a name="resnet50-ssd"></a>ResNet50-SSD
Image classification trained on ImageNet.

[Liu, et al.](https://arxiv.org/pdf/1512.02325v4.pdf) "...a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales
per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes."

### Start server
```bash
mxnet-model-server --models SSD=https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model
```

### Run prediction
```bash
wget https://www.dphotographer.co.uk/users/21963/thm1024/1337890426_Img_8133.jpg
curl -X POST http://127.0.0.1:8080/SSD/predict -F "data=@1337890426_Img_8133.jpg"
```


## <a name="resnext101"></a>ResNext101-64x4d
Image classification trained on ImageNet.

[Xie, et al.](https://arxiv.org/pdf/1611.05431.pdf) "...a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width."

### Start server
```bash
mxnet-model-server --models resnext101=https://s3.amazonaws.com/model-server/models/resnext-101-64x4d/resnext-101-64x4d.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/resnext101/predict -F "data=@kitten.jpeg"
```


## <a name="squeezenet"></a>SqueezeNet v1.1
Image classification trained on ImageNet.

[Iandola, et al.](https://arxiv.org/pdf/1602.07360v4.pdf) "... SqueezeNet achieves
AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally,
with model compression techniques, we are able to compress SqueezeNet to less
than 0.5MB (510× smaller than AlexNet)."

### Start server
```bash
mxnet-model-server --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "data=@kitten.jpeg"
```


## <a name="vgg16"></a>VGG16
Image classification trained on ImageNet.

[Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf) "... the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19
weight layers."

### Start server
```bash
mxnet-model-server --models vgg16=https://s3.amazonaws.com/model-server/models/vgg16/vgg16.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/vgg16/predict -F "data=@kitten.jpeg"
```


## <a name="vgg19"></a>VGG19
Image classification trained on ImageNet.

[Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf) "... the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19
weight layers."

### Start server
```bash
mxnet-model-server --models vgg19=https://s3.amazonaws.com/model-server/models/vgg19/vgg19.model
```

### Run prediction
```bash
curl -X POST http://127.0.0.1:8080/vgg19/predict -F "data=@kitten.jpeg"
```
