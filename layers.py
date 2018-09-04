#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 22:03:24 2018

@author: valaxw
"""
from keras.layers import Activation, Add, Conv2D, BatchNormalization
from keras.regularizers import l2


class resnetv1:
    def conv_block(input_tensor, kernel_size, filters, stage, block, strides = (2,2)): 
            filters1, filters2, filters3 = filters
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'
        
            x = Conv2D(filters1, (1, 1), strides = strides, kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2a')(input_tensor)
            x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)
        
            x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2b')(x)
            x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
            x = Activation('relu')(x)
        
            x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', use_bias = False, kernel_regularizer = l2(1e-4), name=conv_name_base + '2c')(x)
            x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        
            shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer = l2(1e-4),
                                     name=conv_name_base + '1')(input_tensor)
            shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
        
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            return x
        
    def identity_block(input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Conv2D(filters1, (1, 1), kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

class resnetv2():
    def conv_block(input_tensor, kernel_size, filters, stage, block, strides = (2,2)): #https://arxiv.org/pdf/1603.05027.pdf
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2a')(x)
        
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2b')(x)
        
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2c')(x)
    
        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer = l2(1e-4),
                                 name=conv_name_base + '1')(input_tensor)
        x = Add()([x, shortcut])
        return x

    def identity_block(input_tensor, kernel_size, filters, stage, block): #https://arxiv.org/pdf/1603.05027.pdf
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters1, (1, 1), kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2a')(x)
        
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2b')(x)
        
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False, name=conv_name_base + '2c')(x)
        
        x = Add()([x, input_tensor])
        return x
        
def residual_block(input_tensor, depth, filters, stage):
    conv_name_base = 'res' + str(stage) + '-0_branch'
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    
    for i in range(depth-1):
        conv_name_base = 'res' + str(stage) + '-' + str(i) + 'a_branch'
        x1 = Conv2D(filters, (3, 3), activation = 'relu', padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), name=conv_name_base)(x)
        conv_name_base = 'res' + str(stage) + '-' + str(i) + 'b_branch'
        x2 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), name=conv_name_base)(x1)
        x = Add()([x, x2])
    
    conv_name_base = 'res' + str(stage) + '-' + 'last_branch'
    x = Conv2D(3, (3, 3), activation = 'relu', padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), name=conv_name_base)(x)    
    x = Add()([x, input_tensor])
    return x

