# Model Zoo
# Introduction:
This is a side project where Python Scripts implementing different Convolutional Neural Network Architectures are pre-written to aid in future projects. Almost all of the scripts are not faithful replications of the proposed models; however, they are largely similar and (theoretically) should work similarly. There are also additional modifications made to some architectures to improve accuracy. Keras deep learning library is used.  

# How To Use:
## VGG:
Features:
- Filter Size: Instead of the large filter size used by LeNet (9x9 or 11x11), VGG uses a much smaller filter size of 3x3. Despite the smaller receptive field offered by smaller filters, VGG compensates by stacking Convolutional Layers on top of each other. 
- High Learning Ability: Through sheer depth of the network, VGG is capable of learning various different features of images. This allows it to be a very good feature extractor

link: https://arxiv.org/pdf/1409.1556.pdf

### VGG16:
```python
from VGG import VGG16
VGG16_in = Input(shape = (299, 299, 3))
model = VGG16(VGG16_in, ratio = 1, num_class = 1000, dropout = 0.5)
```
where ratio is the how wide the model should be. Ratio = 1 means that the model will be as wide as the proposed model in the original paper, ratio = 2 means half as wide, and so on and so forth. 

### VGG19:
```python
from VGG import VGG19
VGG19_in = Input(shape = (299, 299, 3))
model = VGG19(VGG19_in, ratio = 1, num_class = 1000, dropout = 0.5)
```
### VGG_modified:
This is a modification of the VGG19 architecture. Instead of using fully-connected layers at the top of the architecture, GlobalAveragePooling is used instead. This is inspired from the concept elaborated in the Network-in-Network (NiN) paper (link: https://arxiv.org/pdf/1512.00567.pdf) 
```python
from VGG import VGG_modified
VGG_mod_in = Input(shape = (299, 299, 3))
model = VGG_modified(VGG_mod_in, ratio = 1, num_class = 1000, dropout = 0.5)
```

## Inception (V1 - V3)
Features (InceptionV2, not applied):
- Different Filter Sizes: An inception block has multiple filter sizes (1x1, 3x3, and 5x5) running in parallel. The rationale is that since different filter sizes offer different benefits in identify features, thus multiple sizes should be used instead of only one. 
- Bottleneck layers: A Convolutional layer with filter size 1x1 is first applied before the image is passed to the 3x3 and 5x5 convolution. The bottleneck decreases number of features, keeping only the most relevant ones, before being passed through the costly 3x3 and 5x5 convolutional layers. The authors took inspiratoin from NiN
- GlobalAveragePooling: See explanation above (VGG_Modified)
- Batch Normalization: The BN layer computes the mean and standard deviation of the input feature maps, and then normalizes it by subtracting the mean and dividing the standard deviation. This makes training faster as the next layer will be less affected by the previous layers as the means and variance will be fixed. 
- Factorization: 5x5 filters are replaced with consecutive 3x3 filters. This increases efficiency while achieving the same results. 

### InceptionV3:
Features:
- Features of InceptionV2
- Further Factorization: By flattening the already-small 3x3 filters, efficiency of the model can be improved. This is achieved through asymmetric filters. For example, a 3x3 filter can be decomposed into a 1x3 filter followed by a 3x1 filter. This decreases the number of operations. 

```python
from InceptionV3 import inceptionv3
inception_in = Input(shape = (299, 299, 3))
model = inceptionv3(inception_in, ratio = 1, num_A = 3, num_B = 4, num_C = 2, num_class = 1000, dropout = 0.5)
```
link: https://arxiv.org/pdf/1512.00567.pdf

## ResNet
Features: 
- Skip-Connections: This allows the model to learn identity mappings more easily, thus allowing models to go much deeper without suffering saturation and degradation of accuracy.

### ResNet V1
- Bottleneck Layer: 3x (Conv2D --> Batch_Normalization --> Activation)
- Activation for last layer placed after the skip-connection.

```python
from resnet import ResNet
resnetv1_in = Input(shape = (299, 299, 3))
model = ResNet(resnetv1_in, depth = 50, num_classes = 1000, model_type = 'v1')
```
link: https://arxiv.org/pdf/1512.03385.pdf

### ResNet V2
- Bottleneck Layer: 3x (Batch_Normalization --> Activation --> Conv2D)
- No activation after skip-connection.
```python
from resnet import ResNet
resnetv2_in = Input(shape = (299, 299, 3))
model = ResNet(resnetv2_in, depth = 50, num_classes = 1000, model_type = 'v2')
```
link: https://arxiv.org/pdf/1603.05027.pdf

## Inception-v4 and Inception-ResNet 
### InceptionV4
- Improved Inceptionv3 architecture for faster training speed and better performance
- Hyperparameters are updated

```python
from InceptionV4 import Inceptionv4
Inceptionv4_in = Input(shape = (299, 299, 3))
model = Inceptionv4(Inceptionv4_in, ratio = 1, num_A = 4, num_B = 7, num_C = 3, num_classes = 1000, dropout = 0.8)
```
link: https://arxiv.org/pdf/1602.07261.pdf


### Inception-ResNet-v1 and Inception-ResNet-v2
- Addition of Skip-Connection inspired by ResNet-v1

```python
from Inception-ResNet import Inception_ResNet
Inception_ResNet_in = Input(shape = (299, 299, 3))
model = Inception_ResNet(Inception_ResNet_in, ratio = 1, num_A = 5, num_B = 10, num_C = 5, num_classes = 1000, dropout = 0.8, model_type = 'v2')

'''
or v1:
model = Inception_ResNet(Inception_ResNet_in, ratio = 1, num_A = 5, num_B = 10, num_C = 5, num_classes = 1000, dropout = 0.8, model_type = 'v1')
'''
```

link: https://arxiv.org/pdf/1602.07261.pdf

## Xception
[To be Added]

## MobileNet
[To be Added]

## DenseNet
[To be Added]

## EffNet
[To be Added]

## SqueezeNet
[To be Added]

## NASNet
[To be Added]

## SimpNet
[To be Added]
