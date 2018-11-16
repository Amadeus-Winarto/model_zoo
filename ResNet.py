#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:27:28 2018

@author: valaxw
"""
from keras.models import Model
from keras.layers import Dense, Activation, Add
from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2

def ResNet(model_input, depth, num_classes, model_type = 'v2'): #ResNet Inspired Architecture
    if model_type == 'v1':
        block_depth = 2
        filters = 64
        
        x = Conv2D(64, (3, 3), strides = 2, padding = 'valid', kernel_initializer='he_normal')(model_input)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        
        for i in range(depth):
            block_name = 'a'
            filter_list = [filters, filters, filters *4]
            
            x = resnetv1.conv_block(x, 3, filter_list , stage = block_depth, block = block_name, strides = (1, 1))
            
            if block_depth < 6:
                block_depth += 1
                
            for i in range(block_depth-1):
                block_name = chr(ord(block_name)+1)
                
                x = resnetv1.identity_block(x, 3, filter_list , stage = block_depth, block = block_name)
            
            if filters < 512:
                filters = filters * 2
            x = Conv2D(filters, (3, 3), padding = 'valid', kernel_initializer='he_normal')(x)
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
    
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(num_classes, activation='softmax', name='fc')(x)
        
        model = Model(model_input, x)        
        return model
    
    elif model_type == 'v2':
        block_depth = 2
        filters = 64
        
        x = Conv2D(64, (3, 3), strides = 2, padding = 'valid', kernel_initializer='he_normal')(model_input)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        
        for i in range(depth):
            block_name = 'a'                
            filter_list = [filters, filters, filters *4]
            
            x = resnetv2.conv_block(x, 3, filter_list , stage = block_depth, block = block_name, strides = (1, 1))
            
            if block_depth < 6:
                block_depth += 1
                
            for i in range(block_depth-1):
                block_name = chr(ord(block_name)+1)
                
                x = resnetv2.identity_block(x, 3, filter_list , stage = block_depth, block = block_name)
            
            if filters < 512:
                filters = filters * 2
            x = Conv2D(filters, (3, 3), strides = 2, padding = 'valid', kernel_initializer='he_normal')(x)
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(num_classes, activation='softmax', name='fc')(x)
        
        model = Model(model_input, x)        
        return model

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
        #conv_name_base = 'res' + str(stage) + block + '_branch'
        #bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = BatchNormalization(axis=3)(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False)(x)
        
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False)(x)
        
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False)(x)
    
        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(input_tensor)
        x = Add()([x, shortcut])
        return x

    def identity_block(input_tensor, kernel_size, filters, stage, block): #https://arxiv.org/pdf/1603.05027.pdf
        filters1, filters2, filters3 = filters
        #conv_name_base = 'res' + str(stage) + block + '_branch'
        #bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = BatchNormalization(axis=3)(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters1, (1, 1), kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False)(x)
        
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False)(x)
        
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bias = False)(x)
        
        x = Add()([x, input_tensor])
        return x
