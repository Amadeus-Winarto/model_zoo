#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:09:13 2018

@author: valaxw
"""
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Concatenate, Dropout
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.regularizers import l2

def Inceptionv4(model_input, ratio = 1, num_A = 4, num_B = 7, num_C = 3, num_classes = 1000, dropout = 0.8):
    stem = inceptionv4.Stem(model_input, ratio)
    moduleA = inceptionv4.ModuleA(stem, ratio)
    for i in range(num_A -1):
        moduleA = inceptionv4.ModuleA(moduleA, ratio)
        
    reductionA = inceptionv4.ReductionA(moduleA, ratio)
    
    moduleB = inceptionv4.ModuleB(reductionA, ratio)
    for i in range(num_B -1):
        moduleB = inceptionv4.ModuleB(moduleB, ratio)
        
    reductionB = inceptionv4.ReductionB(moduleB, ratio)
    
    moduleC = inceptionv4.ModuleC(reductionB, ratio)
    for i in range(num_C -1):
        moduleC = inceptionv4.ModuleC(moduleC, ratio)
        
    x = GlobalAveragePooling2D(name='avg_pool')(moduleC)
    x = Dropout(dropout)(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)
    
    model = Model(model_input, x)
    return model

class layers:
    def conv(layer_input, filter_num = 32, filter_size = (3, 3), strides = 1, use_bias = False, padding = 'same', kernel_initializer='he_normal', kernel_regularizer = l2(1e-4), use_bn = 'True', activation = 'relu'):
        convx = Conv2D(filters = filter_num, kernel_size = filter_size, strides = strides, use_bias = use_bias, padding = padding, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(layer_input)
        if use_bn == 'True':
            convx = BatchNormalization(scale=False)(convx)
        convx = Activation(activation)(convx)
        return convx
    
class inceptionv4:
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
        pool4 = MaxPooling2D(pool_size=(3, 3), strides = (1,1), padding='valid')(concat3)
        concat4 = Concatenate()([conv4_1, pool4])
        
        return concat4
    
    def ModuleA(block_input, ratio):
        avgpool = AveragePooling2D(pool_size=(3, 3), strides = (1,1), padding='same')(block_input)
        conv1_1 = layers.conv(avgpool, 96 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 96 // ratio, (1, 1))
        
        conv3_1 = layers.conv(block_input, 64 // ratio, (1, 1))
        conv3_2 = layers.conv(conv3_1, 96 // ratio, (3, 3))
        
        conv4_1 = layers.conv(block_input, 64 // ratio, (1, 1))
        conv4_2 = layers.conv(conv4_1, 96 // ratio, (3, 3))
        conv4_3 = layers.conv(conv4_2, 96 // ratio, (3, 3))
        
        concat1 = Concatenate()([conv1_1, conv2_1, conv3_2, conv4_3])
        return concat1
    
    def ModuleB(block_input, ratio):
        avgpool = AveragePooling2D(pool_size=(3, 3), strides = (1,1), padding='same')(block_input)
        conv1_1 = layers.conv(avgpool, 128 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 384 // ratio, (1, 1))
        
        conv3_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        conv3_2 = layers.conv(conv3_1, 224 // ratio, (1, 7))
        conv3_3 = layers.conv(conv3_2, 256 // ratio, (1, 7))
        
        conv4_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        conv4_2 = layers.conv(conv4_1, 192 // ratio, (1, 7))
        conv4_3 = layers.conv(conv4_2, 224 // ratio, (7, 1))
        conv4_4 = layers.conv(conv4_3, 224 // ratio, (1, 7))
        conv4_5 = layers.conv(conv4_4, 256 // ratio, (7, 1))
        
        concat1 = Concatenate()([conv1_1, conv2_1, conv3_3, conv4_5])
        return concat1
    
    def ModuleC(block_input, ratio):
        avgpool = AveragePooling2D(pool_size=(3, 3), strides = (1,1), padding='same')(block_input)
        conv1_1 = layers.conv(avgpool, 256 // ratio, (1, 1))
        
        conv2_1 = layers.conv(block_input, 256 // ratio, (1, 1))
        
        conv3_1 = layers.conv(block_input, 384 // ratio, (1, 1))
        conv3a_2 = layers.conv(conv3_1, 256 // ratio, (1, 3))
        conv3b_2 = layers.conv(conv3_1, 256 // ratio, (3, 1))
        
        conv4_1 = layers.conv(block_input, 384 // ratio, (1, 1))
        conv4_2 = layers.conv(conv4_1, 448 // ratio, (1, 3))
        conv4_3 = layers.conv(conv4_2, 512 // ratio, (3, 1))
        conv4a_4 = layers.conv(conv4_3, 256 // ratio, (3, 1))
        conv4b_4 = layers.conv(conv4_3, 256 // ratio, (1, 3))
        
        concat1 = Concatenate()([conv1_1, conv2_1, conv3a_2, conv3b_2, conv4a_4, conv4b_4])
        return concat1
        
    def ReductionA(block_input, ratio):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(block_input)
        
        conv2_1 = layers.conv(block_input, 96 // ratio, (3, 3), strides = 2, padding='valid')
        
        
        conv3_1 = layers.conv(block_input, 192, (1, 1))
        conv3_2 = layers.conv(conv3_1, 224 // ratio, (3, 3))
        conv3_3 = layers.conv(conv3_2, 256 // ratio, (3, 3), strides = 2, padding='valid')
        
        concat1 = Concatenate()([maxpool, conv2_1, conv3_3])
        return concat1
    
    def ReductionB(block_input, ratio):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding='valid')(block_input)
        
        conv2_1 = layers.conv(block_input, 192 // ratio, (1, 1))
        conv2_2 = layers.conv(conv2_1, 192 // ratio, (3, 3), strides = 2, padding='valid')
        
        conv3_1 = layers.conv(block_input, 256 // ratio, (1, 1))
        conv3_2 = layers.conv(conv3_1, 256 // ratio, (1, 7))
        conv3_3 = layers.conv(conv3_2, 320 // ratio, (7, 1))
        conv3_4 = layers.conv(conv3_3, 320 // ratio, (3, 3), strides = 2, padding='valid')
        
        concat1 = Concatenate()([maxpool, conv2_2, conv3_4])
        return concat1
    
        
        
        
        
        
        
        
        
    