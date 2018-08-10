# model_zoo
A list of Python Scripts that implement various model architectures in Keras

# How To Use:
## InceptionV3:
import models

inception_in = Input()

inception = inceptionv3(inception_in, ratio = 1, num_A = 3, num_B = 4, num_C = 2, num_class = 1000, lr = 1e-5)
