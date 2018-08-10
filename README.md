# model_zoo
A list of Python Scripts that implement various model architectures in Keras

# How To Use:
## InceptionV3:
```python
from InceptionV3 import inceptionv3
inception_in = Input(shape = (299, 299, 3)
inception = inceptionv3(inception_in, ratio = 1, num_A = 3, num_B = 4, num_C = 2, num_class = 1000, lr = 1e-5
```
where ratio is the how wide the model should be. Ratio = 1 means that the model will be as wide as the proposed model in the original paper, ratio = 2 means half as wide, and so on and so forth. 
link: https://arxiv.org/pdf/1512.00567.pdf
