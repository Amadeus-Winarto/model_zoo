# model_zoo
A list of Python Scripts that implement various model architectures in Keras

# Introduction:
This is a side project where scripts implementing different Convolutional Neural Network Architectures are pre-written to aid in future projects. Almost all of the scripts are not faithful replications of the proposed models; however, they are largely similar and (theoretically) should work similarly. There are also additional modifications made to some architectures to improve accuracy. 

# How To Use:
## VGG:
### VGG16:
```python
from VGG import VGG16
VGG16_in = Input(shape = (299, 299, 3)
model = VGG16(VGG16_in, ratio = 1, num_class = 1000, lr = 1e-5)
```
where ratio is the how wide the model should be. Ratio = 1 means that the model will be as wide as the proposed model in the original paper, ratio = 2 means half as wide, and so on and so forth. 
link: https://arxiv.org/pdf/1409.1556.pdf

### VGG19:
```python
from VGG import VGG19
VGG19_in = Input(shape = (299, 299, 3)
model = VGG19(VGG19_in, ratio = 1, num_class = 1000, lr = 1e-5)
```
link: https://arxiv.org/pdf/1409.1556.pdf

### VGG_modified:
This is a modification of the VGG19 architecture. Instead of using fully-connected layers at the top of the architecture, GlobalAveragePooling is used instead. This is inspired from the concept elaborated in the Network-in-Network paper (link: https://arxiv.org/pdf/1512.00567.pdf) 
```python
from VGG import VGG_modified
VGG-mod_in = Input(shape = (299, 299, 3)
model = VGG_modified(VGG-mod_in, ratio = 1, num_class = 1000, lr = 1e-5)
```
link: https://arxiv.org/pdf/1409.1556.pdf


## InceptionV3:
```python
from InceptionV3 import inceptionv3
inception_in = Input(shape = (299, 299, 3)
inception = inceptionv3(inception_in, ratio = 1, num_A = 3, num_B = 4, num_C = 2, num_class = 1000, lr = 1e-5
```
link: https://arxiv.org/pdf/1512.00567.pdf

## InceptionV4

## Inception-ResNet
