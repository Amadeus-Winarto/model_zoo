#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:03:59 2018

@author: valaxw
"""

from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Concatenate, Dropout, Add
from keras.layers import GlobalAveragePooling2D, MaxPooling2D
from keras.regularizers import l2

def Inception_ResNet(model_input, ratio = 1, num_A = 5, num_B = 10, num_C = 5, num_classes = 1000, lr = 1e-5, dropout = 0.8, model_type = 'v2'):
    if model_type == 'v1':
        stem = inception_ResNet_v1.Stem(model_input, ratio)
        moduleA = inception_ResNet_v1.ModuleA(stem, ratio)
        for i in range(num_A -1):
            moduleA = inception_ResNet_v1.ModuleA(moduleA, ratio)
            
        reductionA = inception_ResNet_v1.ReductionA(moduleA, ratio)
        
        moduleB = inception_ResNet_v1.ModuleB(reductionA, ratio)
        for i in range(num_B -1):
            moduleB = inception_ResNet_v1.ModuleB(moduleB, ratio)
            
        reductionB = inception_ResNet_v1.ReductionB(moduleB, ratio)
        
        moduleC = inception_ResNet_v1.ModuleC(reductionB, ratio)
        for i in range(num_C -1):
            moduleC = inception_ResNet_v1.ModuleC(moduleC, ratio)
            
        x = GlobalAveragePooling2D(name='avg_pool')(moduleC)
        x = Dropout(dropout)(x)
        x = Dense(num_classes, activation='softmax', name='fc')(x)
        
        model = Model(model_input, x)
        return model
    if model_type == 'v2':
        stem = inception_ResNet_v2.Stem(model_input, ratio)
        moduleA = inception_ResNet_v2.ModuleA(stem, ratio)
        for i in range(num_A -1):
            moduleA = inception_ResNet_v2.ModuleA(moduleA, ratio)
            
        reductionA = inception_ResNet_v2.ReductionA(moduleA, ratio)
        
        moduleB = inception_ResNet_v2.ModuleB(reductionA, ratio)
        for i in range(num_B -1):
            moduleB = inception_ResNet_v2.ModuleB(moduleB, ratio)
            
        reductionB = inception_ResNet_v2.ReductionB(moduleB, ratio)
        
        moduleC = inception_ResNet_v2.ModuleC(reductionB, ratio)
        for i in range(num_C -1):
            moduleC = inception_ResNet_v2.ModuleC(moduleC, ratio)
            
        x = GlobalAveragePooling2D(name='avg_pool')(moduleC)
        x = Dropout(dropout)(x)
        x = Dense(num_classes, activation='softmax', name='fc')(x)
        
        model = Model(model_input, x)
        return model
    
class layers:
    def conv(layer_input, filter_num = 32, filter_size = (3, 3), strides = 1, use_bias = False, padding = 'valid', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bn = 'True', activation = 'relu'):
        convx = Conv2D(filter_num, filter_size, strides, use_bias, padding, kernel_initializer, kernel_regularizer)(layer_input)
        if use_bn == 'True':
            convx = BatchNormalization(scale=False)(convx)
        if activation != 'None':
            convx = Activation(activation)(convx)
        
        return convx

class inception_ResNet_v1:
    def Stem(img_input, ratio):
        conv1 = layers.conv(img_input, 32 // ratio, (3,3), strides = 2, padding = 'valid')
        conv2 = layers.conv(conv1, 32 // ratio, (3,3), padding = 'valid')
        conv3 = layers.conv(conv2, 64 // ratio, (3,3))
        pool1 = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(conv3)
        conv5 = layers.conv(pool1, 80 // ratio, (1,1))
        conv6 = layers.conv(conv5, 192 // ratio, (3,3), padding = 'valid')
        conv7 = layers.conv(conv6, 256 // ratio, (3,3), strides = 2, padding = 'valid')
        return conv7
    
    def ModuleA(block_input, ratio):
        conv1_1 = layers.conv(block_input, 32 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 32 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 32 // ratio, (3, 3))
        
        conv3_1 = layers.conv(block_input, 32 // ratio, (1, 1))
        conv3_2 = layers.conv(conv3_1, 32 // ratio, (3, 3))
        conv3_3 = layers.conv(conv3_2, 32 // ratio, (3, 3))        
        
        concat1 = Concatenate()([conv1_1, conv2_2, conv3_3])
        conv4 = layers.conv(concat1, 256 // ratio, (1, 1), activation = 'None')
        
        add = Add()([block_input, conv4])
        return add
    
    def ModuleB(block_input, ratio):
        conv1_1 = layers.conv(block_input, 128 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 128 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 128 // ratio, (1, 7))
        conv2_3 = layers.conv(conv2_2, 128 // ratio, (7, 1))
        
        concat1 = Concatenate()([conv1_1, conv2_3])
        conv4 = layers.conv(concat1, 896 // ratio, (1, 1), activation = 'None')
        
        add = Add()([block_input, conv4])
        return add
    
    def ModuleC(block_input, ratio):
        conv1_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 192 // ratio, (1, 3))
        conv2_3 = layers.conv(conv2_2, 192 // ratio, (3, 1))
        
        concat1 = Concatenate()([conv1_1, conv2_3])
        conv4 = layers.conv(concat1, 1792 // ratio, (1, 1), activation = 'None')
        
        add = Add()([block_input, conv4])
        return add
        
    def ReductionA(block_input, ratio):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(block_input)
        
        conv2_1 = layers.conv(block_input, 96 // ratio, (3, 3), strides = 2, padding='valid')
        
        
        conv3_1 = layers.conv(block_input, 192, (1, 1))
        conv3_2 = layers.conv(conv3_1, 192 // ratio, (3, 3))
        conv3_3 = layers.conv(conv3_2, 256 // ratio, (3, 3), strides = 2, padding='valid')
        
        concat1 = Concatenate()([maxpool, conv2_1, conv3_3])
        return concat1
    
    def ReductionB(block_input, ratio):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(block_input)
        
        conv2_1 = layers.conv(block_input, 256 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 384 // ratio, (3, 3), strides = 2, padding='valid')
        
        conv3_1 = layers.conv(block_input, 256 // ratio, (1, 1))
        conv3_2 = layers.conv(conv3_1, 256 // ratio, (3, 3), strides = 2, padding='valid')
        
        conv4_1 = layers.conv(block_input, 256 // ratio, (1, 1))
        conv4_2 = layers.conv(conv4_1, 256 // ratio, (3, 3))
        conv4_3 = layers.conv(conv4_2, 256 // ratio, (3, 3), strides = 2, padding='valid')
        
        concat1 = Concatenate()([maxpool, conv2_2, conv3_2, conv4_3])
        return concat1
    

class inception_ResNet_v2:
    def Stem(img_input, ratio):
        conv1_1 = Conv2D(32 // ratio, (3, 3), strides = 2, use_bias = False, padding = 'valid', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(img_input)
        conv1_1 = BatchNormalization(scale=False)(conv1_1)
        conv1_1 = Activation('relu')(conv1_1)
        conv1_2 = Conv2D(32// ratio, (3, 3), strides = 1, use_bias = False, padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(conv1_1)
        conv1_2 = BatchNormalization(scale=False)(conv1_2)
        conv1_2 = Activation('relu')(conv1_2)
        conv1_3 = Conv2D(64// ratio, (3, 3), strides = 1, use_bias = False, padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(conv1_2)
        conv1_3 = BatchNormalization(scale=False)(conv1_3)
        conv1_3 = Activation('relu')(conv1_3)
        
        conv2_1 = Conv2D(96// ratio, (3, 3), strides = 2, use_bias = False, padding = 'valid', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(conv1_3)
        conv2_1 = BatchNormalization(scale=False)(conv2_1)
        conv2_1 = Activation('relu')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(conv1_3)
        concat2 = Concatenate()([conv2_1, pool2])
        
        conv3a_1 = Conv2D(64// ratio, (1, 1), strides = 1, use_bias = False, padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(concat2)
        conv3a_1 = BatchNormalization(scale=False)(conv3a_1)
        conv3a_1 = Activation('relu')(conv3a_1)
        conv3a_2 = Conv2D(96// ratio, (3, 3), strides = 1, use_bias = False, padding = 'valid', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(conv3a_1)
        conv3a_2 = BatchNormalization(scale=False)(conv3a_2)
        conv3a_2 = Activation('relu')(conv3a_2)
        conv3b_1 = Conv2D(64// ratio, (1, 1), strides = 1, use_bias = False, padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(concat2)
        conv3b_1 = BatchNormalization(scale=False)(conv3b_1)
        conv3b_1 = Activation('relu')(conv3b_1)
        conv3b_2 = Conv2D(64// ratio, (7, 1), strides = 1, use_bias = False, padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(conv3b_1)
        conv3b_2 = BatchNormalization(scale=False)(conv3b_2)
        conv3b_2 = Activation('relu')(conv3b_2)
        conv3b_3 = Conv2D(64// ratio, (1, 7), strides = 1, use_bias = False, padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(conv3b_2)
        conv3b_3 = BatchNormalization(scale=False)(conv3b_3)
        conv3b_3 = Activation('relu')(conv3b_3)
        conv3b_4 = Conv2D(96// ratio, (3, 3), strides = 1, use_bias = False, padding = 'valid', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(conv3b_3)
        conv3b_4 = BatchNormalization(scale=False)(conv3b_4)
        conv3b_4 = Activation('relu')(conv3b_4)
        concat3 = Concatenate()([conv3a_2, conv3b_4])
        
        conv4_1 = Conv2D(192// ratio, (3, 3), strides = 1, use_bias = False, padding = 'valid', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4))(concat3)
        conv4_1 = BatchNormalization(scale=False)(conv4_1)
        conv4_1 = Activation('relu')(conv4_1)
        pool4 = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(concat3)
        concat4 = Concatenate()([conv4_1, pool4])
        
        return concat4
    
    def ModuleA(block_input, ratio):
        conv1_1 = layers.conv(block_input, 32 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 32 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 32 // ratio, (3, 3))
        
        conv3_1 = layers.conv(block_input, 32 // ratio, (1, 1))
        conv3_2 = layers.conv(conv3_1, 48 // ratio, (3, 3))
        conv3_3 = layers.conv(conv3_2, 64 // ratio, (3, 3))        
        
        concat1 = Concatenate()([conv1_1, conv2_2, conv3_3])
        conv4 = layers.conv(concat1, 384 // ratio, (1, 1), activation = 'None')
        
        add = Add()([block_input, conv4])
        return add
    
    def ModuleB(block_input, ratio):
        conv1_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 128 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 160 // ratio, (1, 7))
        conv2_3 = layers.conv(conv2_2, 192 // ratio, (7, 1))
        
        concat1 = Concatenate()([conv1_1, conv2_3])
        conv4 = layers.conv(concat1, 1154 // ratio, (1, 1), activation = 'None')
        
        add = Add()([block_input, conv4])
        return add
    
    def ModuleC(block_input, ratio):
        conv1_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 224 // ratio, (1, 3))
        conv2_3 = layers.conv(conv2_2, 256 // ratio, (3, 1))
        
        concat1 = Concatenate()([conv1_1, conv2_3])
        conv4 = layers.conv(concat1, 2048 // ratio, (1, 1), activation = 'None')
        
        add = Add()([block_input, conv4])
        return add
    
    def ReductionA(block_input, ratio):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(block_input)
        
        conv2_1 = layers.conv(block_input, 96 // ratio, (3, 3), strides = 2, padding='valid')
        
        conv3_1 = layers.conv(block_input, 256, (1, 1))
        conv3_2 = layers.conv(conv3_1, 256 // ratio, (3, 3))
        conv3_3 = layers.conv(conv3_2, 384 // ratio, (3, 3), strides = 2, padding='valid')
        
        concat1 = Concatenate()([maxpool, conv2_1, conv3_3])
        return concat1
    
    def ReductionB(block_input, ratio):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(block_input)
        
        conv2_1 = layers.conv(block_input, 256, (1, 1))
        conv2_2 = layers.conv(conv2_1, 384 // ratio, (3, 3), strides = 2, padding='valid')
        
        conv3_1 = layers.conv(block_input, 256, (1, 1))
        conv3_2 = layers.conv(conv3_1, 288 // ratio, (3, 3), strides = 2, padding='valid')
        
        conv4_1 = layers.conv(block_input, 256, (1, 1))
        conv4_2 = layers.conv(conv4_1, 288 // ratio, (3, 3))
        conv4_3 = layers.conv(conv4_2, 320 // ratio, (3, 3), strides = 2, padding='valid')
        
        concat1 = Concatenate()([maxpool, conv2_2, conv3_2, conv4_3])
        return concat1
        

    
        
        
        
        
        
        
        
        
    