#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:21:34 2018

@author: valaxw
"""
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Concatenate, Dropout, Add
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import ReLU, ELU, LeakyReLU
from keras.regularizers import l2

from ResNet import layers

def ResNetv2(img_input, ratio = 1, num_A = 2, num_B = 2, num_C = 2, num_D = 2, activation_type = 'relu', pool = 'max', num_classes = 1000, dropout = 0.5):
    conv1 = Conv2D(64 // ratio, (7, 7), padding = 'same', strides = 2)(img_input)
    pool1 = MaxPooling2D((3,3), strides=(2, 2), padding = 'same')(conv1)
    
    filters = 64 // ratio
    x = resnetv2.identity_block(pool1, filters = filters)
    for i in range(num_A - 1):
        x = resnetv2.identity_block(x, filters = filters)
        
    filters *= 2
    x = resnetv2.conv_block(x, filters = filters)
    for i in range(num_B - 1):
        x = resnetv2.identity_block(x, filters = filters)
    
    filters *= 2
    x = resnetv2.conv_block(x, filters = filters)
    for i in range(num_C - 1):
        x = resnetv2.identity_block(x, filters = filters)
    
    filters *= 2
    x = resnetv2.conv_block(x, filters = filters)
    for i in range(num_D - 1):
        x = resnetv2.identity_block(x, filters = filters)
    
    if pool == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    else:
        x = GlobalMaxPooling2D(name='max_pool')(x)
    x = Dropout(dropout)(x)
    if num_classes == 2:
        x = Dense(1, activation = 'sigmoid', name = 'fc')(x)
    else:
        x = Dense(num_classes, activation='softmax', name='fc')(x)
    
    model = Model(img_input, x)        
    return model

class resnetv2:
    def conv_block(block_input, filters, filter_size = (3, 3), strides = 2,
                   kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'):
        
        x = layers.convV2(block_input, filter_num = filters, filter_size = filter_size, strides = strides, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV2(x, filter_num = filters, filter_size = filter_size, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        
        shortcut = layers.convV2(block_input, filter_num = filters, filter_size = (1, 1), strides = strides, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = 'none')
        x = Add()([x, shortcut])
        return x
    
    def identity_block(block_input, filters, filter_size = (3, 3), 
                       kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'): #https://arxiv.org/pdf/1603.05027.pdf
        
        x = layers.convV2(block_input, filter_num = filters, filter_size = filter_size, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV2(x, filter_num = filters, filter_size = filter_size, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        
        x = Add()([x, block_input])
        return x