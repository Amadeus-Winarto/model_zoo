#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:20:36 2018

@author: valaxw
"""
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Concatenate, Dropout, Add
from keras.layers import GlobalAveragePooling2D, MaxPooling2D
from keras.regularizers import l2


def ResNeXt(model_input, ratio = 1, num_A = 5, num_B = 10, num_C = 5, num_classes = 1000, lr = 1e-5, dropout = 0.8, model_type = 'v2'):
    conv1 = layers.conv(model_input, 64, (7, 7), strides = 2)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='same')(conv1)
    
    
    
    
    model = Model(model_input,)
    return model

class layers:
    def conv(layer_input, filter_num = 32, filter_size = (3, 3), strides = 1, use_bias = False, padding = 'valid', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bn = 'True', activation = 'relu'):
        convx = Conv2D(filter_num, filter_size, strides, use_bias, padding, kernel_initializer, kernel_regularizer)(layer_input)
        if use_bn == 'True':
            convx = BatchNormalization(scale=False)(convx)
        if activation != 'None':
            convx = Activation(activation)(convx)
        
        return convx
    def resnext():
        return convx
    