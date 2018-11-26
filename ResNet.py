#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:21:08 2018

@author: amadeusaw
"""
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Concatenate, Dropout, Add
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import ReLU, ELU, LeakyReLU
from keras.regularizers import l2

def ResNetv1(img_input, ratio = 1, num_A = 3, num_B = 4, num_C = 6, num_D = 3, activation_type = 'relu', num_classes = 1000, dropout = 0.5):
    conv1 = Conv2D(64, (7, 7), padding = 'same', strides = 2)(img_input)
    pool1 = MaxPooling2D((3,3), strides=(2, 2), padding = 'same')(conv1)
    
    filters = (64//ratio, 64// ratio, 256//ratio)
    x = resnetv1.conv_block(pool1, filters = filters, strides = 1)
    for i in range(num_A - 1):
        x = resnetv1.identity_block(x, filters = filters)
        
    filters = tuple([2 * j for j in filters])
    x = resnetv1.conv_block(x, filters = filters)
    for i in range(num_B - 1):
        x = resnetv1.identity_block(x, filters = filters)
    
    filters = tuple([2 * j for j in filters])
    x = resnetv1.conv_block(x, filters = filters)
    for i in range(num_C - 1):
        x = resnetv1.identity_block(x, filters = filters)
    
    filters = tuple([2 * j for j in filters])
    x = resnetv1.conv_block(x, filters = filters)
    for i in range(num_D - 1):
        x = resnetv1.identity_block(x, filters = filters)
    
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout)(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)
    
    model = Model(img_input, x)        
    return model

def ResNetv2(img_input, ratio = 1, num_A = 3, num_B = 4, num_C = 6, num_D = 3, num_classes = 1000, dropout = 0.5):
    conv1 = Conv2D(64, (7, 7), padding = 'same', strides = 2)(img_input)
    pool1 = MaxPooling2D((3,3), strides=(2, 2), padding = 'same')(conv1)
    
    filters = (64//ratio, 64// ratio, 256//ratio)
    x = resnetv2.conv_block(pool1, filters = filters, strides = 1)
    for i in range(num_A - 1):
        x = resnetv2.identity_block(x, filters = filters)
        
    filters = tuple([2 * j for j in filters])
    x = resnetv2.conv_block(x, filters = filters)
    for i in range(num_B - 1):
        x = resnetv2.identity_block(x, filters = filters)
    
    filters = tuple([2 * j for j in filters])
    x = resnetv2.conv_block(x, filters = filters)
    for i in range(num_C - 1):
        x = resnetv2.identity_block(x, filters = filters)
    
    filters = tuple([2 * j for j in filters])
    x = resnetv2.conv_block(x, filters = filters)
    for i in range(num_D - 1):
        x = resnetv2.identity_block(x, filters = filters)
    
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout)(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)
    
    model = Model(img_input, x)        
    return model
    
class layers:
    def activation(x, activation_type = 'relu'):
        if activation_type == 'relu':
            out = ReLU()(x)
            return out
        elif activation_type == 'leakyrelu':
            out = LeakyReLU()(x)
            return out
        elif activation_type == 'elu':
            out = ELU()(x)
            return out
        
    def convV1(layer_input, filter_num = 32, filter_size = (3, 3), 
               strides = 1, use_bias = False, padding = 'same', 
               kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'):
        
        
        x = Conv2D(filter_num, filter_size, strides = strides, 
                   padding = padding,
                   kernel_initializer = kernel_initializer,
                   kernel_regularizer = kernel_regularizer,
                   use_bias = use_bias)(layer_input)
        x = BatchNormalization(scale = False)(x)
        x = layers.activation(x, activation_type)
        
        return x
    
    def convV2(layer_input, filter_num = 32, filter_size = (3, 3), 
               strides = 1, use_bias = False, padding = 'same', 
               kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'):
        
        x = BatchNormalization(scale = False)(layer_input)
        x = layers.activation(x, activation_type)
        x = Conv2D(filter_num, filter_size, strides = strides, 
                   padding = padding,
                   kernel_initializer = kernel_initializer,
                   kernel_regularizer = kernel_regularizer,
                   use_bias = use_bias)(x)
        return x

class resnetv1:
    def conv_block(block_input, filters, filter_size = (3, 3), kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'):
        
        filters1, filters2, filters3 = filters
        x = layers.convV1(block_input, filter_num = filters1, filter_size = (1, 1), strides = 2, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV1(x, filter_num = filters2, filter_size = filter_size, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV1(x, filter_num = filters3, filter_size = (1,1), kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
    
        shortcut = Conv2D(filters3, (1, 1), strides = 2, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(block_input)
        x = Add()([x, shortcut])
        return x
    
    def identity_block(block_input, filters, filter_size = (3, 3), 
                       kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'): #https://arxiv.org/pdf/1603.05027.pdf
        
        filters1, filters2, filters3 = filters        
        x = layers.convV1(block_input, filter_num = filters1, filter_size = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV1(x, filter_num = filters2, filter_size = filter_size, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV1(x, filter_num = filters3, filter_size = (1,1), kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        
        x = Add()([x, block_input])
        return x
    
class resnetv2:
    def conv_block(block_input, filters, filter_size = (3, 3), 
                   strides = 2, kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'):
        
        filters1, filters2, filters3 = filters
        x = layers.convV2(block_input, filter_num = filters1, filter_size = (1, 1), strides = 2, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV2(x, filter_num = filters2, filter_size = filter_size, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV2(x, filter_num = filters3, filter_size = (1,1), kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
    
        shortcut = Conv2D(filters3, (1, 1), strides = 2, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(block_input)
        x = Add()([x, shortcut])
        return x
    
    def identity_block(block_input, filters, filter_size = (3, 3), 
                       kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), activation_type = 'relu'): #https://arxiv.org/pdf/1603.05027.pdf
        
        filters1, filters2, filters3 = filters        
        x = layers.convV2(block_input, filter_num = filters1, filter_size = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV2(x, filter_num = filters2, filter_size = filter_size, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        x = layers.convV2(x, filter_num = filters3, filter_size = (1,1), kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, activation_type = activation_type)
        
        x = Add()([x, block_input])
        return x

