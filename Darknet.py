#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:24:59 2018

@author: valaxw
"""
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Concatenate, Dropout, Add
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import ReLU, ELU, LeakyReLU
from keras.regularizers import l2

def Darknet(img_input, filter_num = 32, num_A = 1, num_B = 2, num_C = 8, num_D = 8, num_E = 4, activation_type = 'relu', regularizer = 1e-4, pool = 'avg', dropout = 0.5, num_classes = 1000, mode = 'v1'):
    x = Conv2D(filter_num, (3,3), padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(regularizer), use_bias = False)(img_input)
    x = BatchNormalization(scale = False)(x)
    x = layers.activation(x, activation_type = activation_type)
    
    x = layers.conv_down(x, filter_num * 2, activation_type = activation_type, regularizer = regularizer, mode = mode)
    for i in range(num_A):
        x = layers.block(x, activation_type = activation_type, filters = (filter_num, filter_num * 2), regularizer = regularizer, mode = mode)
    
    filter_num *= 2        
    x = layers.conv_down(x, filter_num * 2, activation_type = activation_type, regularizer = regularizer, mode = mode)
    for i in range(num_B):
        x = layers.block(x, activation_type = activation_type, filters = (filter_num, filter_num * 2), regularizer = regularizer, mode = mode)
    
    filter_num *= 2        
    x = layers.conv_down(x, filter_num * 2, activation_type = activation_type, regularizer = regularizer, mode = mode)
    for i in range(num_C):
        x = layers.block(x, activation_type = activation_type, filters = (filter_num, filter_num * 2), regularizer = regularizer, mode = mode)
    
    filter_num *= 2        
    x = layers.conv_down(x, filter_num * 2, activation_type = activation_type, regularizer = regularizer, mode = mode)
    for i in range(num_D):
        x = layers.block(x, activation_type = activation_type, filters = (filter_num, filter_num * 2), regularizer = regularizer, mode = mode)
    
    filter_num *= 2        
    x = layers.conv_down(x, filter_num * 2, activation_type = activation_type, regularizer = regularizer, mode = mode)
    for i in range(num_E):
        x = layers.block(x, activation_type = activation_type, filters = (filter_num, filter_num * 2), regularizer = regularizer, mode = mode)
    
    if pool == 'max':
        x = GlobalMaxPooling2D(name = 'max_pool')(x)
    elif pool == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        
    x = Dropout(dropout)(x)
    if num_classes != 2:
        x = Dense(num_classes, activation='softmax', name='fc')(x)
    else:
        x = Dense(num_classes, activation='sigmoid', name='fc')(x)
    
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
    
    def conv_down(layer_input, filters, activation_type = 'relu', regularizer = 1e-4, mode = 'v1'):
        if mode == 'v1':
            x = Conv2D(filters, (3,3), padding = 'same', strides = 2, use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(regularizer))(layer_input)
            x = BatchNormalization(scale = False)(x)
            x = layers.activation(x, activation_type = activation_type)
            return x
        
        elif mode == 'v2':
            x = BatchNormalization(scale = False)(layer_input)
            x = layers.activation(x, activation_type = activation_type)
            x = Conv2D(filters, (3,3), padding = 'same', strides = 2, use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(regularizer))(x)
            return x

    def block(layer_input, activation_type = 'relu', filters = (32, 64), regularizer = 1e-4, mode = 'v1'):
        if mode == 'v1':
            filters1, filters2 = filters
            x = Conv2D(filters1, (1,1), padding = 'same', strides = 1, use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(regularizer))(layer_input)
            x = BatchNormalization(scale = False)(x)
            x = layers.activation(x, activation_type = activation_type)
            
            x = Conv2D(filters2, (3,3), padding = 'same', strides = 1, use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(regularizer))(x)
            x = BatchNormalization(scale = False)(x)
            x = layers.activation(x, activation_type = activation_type)
            
            shortcut = Add()([x, layer_input])
            return shortcut
        elif mode == 'v2':
            filters1, filters2 = filters            
            x = BatchNormalization(scale = False)(layer_input)
            x = layers.activation(x, activation_type = activation_type)
            x = Conv2D(filters1, (1,1), padding = 'same', strides = 1, use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(regularizer))(x)
            
            x = BatchNormalization(scale = False)(layer_input)
            x = layers.activation(x, activation_type = activation_type)
            x = Conv2D(filters2, (3,3), padding = 'same', strides = 1, use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(regularizer))(x)
            
            shortcut = Add()([x, layer_input])
            return shortcut