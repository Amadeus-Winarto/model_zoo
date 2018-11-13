#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 21:09:03 2018

@author: valaxw
"""
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Dense, MaxPooling2D, Flatten, Dropout, Activation, Add, BatchNormalization, Concatenate
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import tensorflow as tf

import os

def moduleA(ratio, input_):
    convA1_11 = Conv2D(64//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convA1_11 = BatchNormalization(scale=False)(convA1_11)
    convA1_11 = Activation('relu')(convA1_11)
    convA1_12 = Conv2D(96//ratio, (3,3), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convA1_11)
    convA1_12 = BatchNormalization(scale=False)(convA1_12)
    convA1_12 = Activation('relu')(convA1_12)
    convA1_13 = Conv2D(96//ratio, (3,3), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convA1_12)
    convA1_13 = BatchNormalization(scale=False)(convA1_13)
    convA1_13 = Activation('relu')(convA1_13)
    
    convA1_21 = Conv2D(48//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convA1_21 = BatchNormalization()(convA1_21)
    convA1_21 = Activation('relu')(convA1_21)
    convA1_22 = Conv2D(64//ratio, (5,5), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convA1_21)
    convA1_22 = BatchNormalization(scale=False)(convA1_22)
    convA1_22 = Activation('relu')(convA1_22)
    
    poolA1 = AveragePooling2D(pool_size=(3, 3), strides = (1,1), padding='same')(input_)
    convA1_31 = Conv2D(64//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(poolA1)
    convA1_31 = BatchNormalization(scale=False)(convA1_31)
    convA1_31 = Activation('relu')(convA1_31)
    
    convA1_41 = Conv2D(64//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convA1_41 = BatchNormalization(scale=False)(convA1_41)
    convA1_41 = Activation('relu')(convA1_41)
    
    concatA1 = Concatenate()([convA1_13, convA1_22, convA1_31, convA1_41])
    
    return concatA1

def resizeA(ratio, input_):
    convA4_1 = Conv2D(384//ratio, (3, 3), strides = (2, 2), use_bias = False, padding = 'valid', kernel_regularizer=l2(0.02))(input_)
    convA4_1 = BatchNormalization(scale=False)(convA4_1)
    convA4_1 = Activation('relu')(convA4_1)

    convA4_2 = Conv2D(64//ratio, (1, 1), strides = (1,1), use_bias = False, padding = 'same', kernel_regularizer=l2(0.02))(input_)
    convA4_2 = BatchNormalization(scale=False)(convA4_2)
    convA4_2 = Activation('relu')(convA4_2)
    convA4_3 = Conv2D(96//ratio, (3, 3), strides = (1,1), use_bias = False, padding = 'same', kernel_regularizer=l2(0.02))(convA4_2)
    convA4_3 = BatchNormalization(scale=False)(convA4_3)
    convA4_3 = Activation('relu')(convA4_3)
    convA4_4 = Conv2D(96//ratio, (3, 3), strides = (2, 2), use_bias = False, padding='valid', kernel_regularizer=l2(0.02))(convA4_3)
    convA4_4 = BatchNormalization(scale=False)(convA4_4)
    convA4_4 = Activation('relu')(convA4_4)

    poolA4 = MaxPooling2D((3, 3), strides=(2, 2))(input_)
    concatA4 = Concatenate()([convA4_1, convA4_4, poolA4])
    
    return concatA4

def moduleB(ratio, input_, num):
    convB1_11 = Conv2D(num//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convB1_11 = BatchNormalization(scale=False)(convB1_11)
    convB1_11 = Activation('relu')(convB1_11)
    convB1_12 = Conv2D(num//ratio, (7,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convB1_11)
    convB1_12 = BatchNormalization(scale=False)(convB1_12)
    convB1_12 = Activation('relu')(convB1_12)
    convB1_13 = Conv2D(num//ratio, (1,7), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convB1_12)
    convB1_13 = BatchNormalization(scale=False)(convB1_13)
    convB1_13 = Activation('relu')(convB1_13)
    convB1_14 = Conv2D(num//ratio, (7,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convB1_13)
    convB1_14 = BatchNormalization(scale=False)(convB1_14)
    convB1_14 = Activation('relu')(convB1_14)
    convB1_15 = Conv2D(192//ratio, (1,7), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convB1_14)
    convB1_15 = BatchNormalization(scale=False)(convB1_15)
    convB1_15 = Activation('relu')(convB1_15)
    
    convB1_21 = Conv2D(num//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convB1_21 = BatchNormalization(scale=False)(convB1_21)
    convB1_21 = Activation('relu')(convB1_21)
    convB1_22 = Conv2D(num//ratio, (1,7), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convB1_21)
    convB1_22 = BatchNormalization(scale=False)(convB1_22)
    convB1_22 = Activation('relu')(convB1_22)
    convB1_23 = Conv2D(192//ratio, (7,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convB1_22)
    convB1_23 = BatchNormalization(scale=False)(convB1_23)
    convB1_23 = Activation('relu')(convB1_23)
    
    poolB1 = AveragePooling2D(pool_size=(3, 3), strides = (1,1), padding='same')(input_)
    convB1_31 = Conv2D(192//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(poolB1)
    convB1_31 = BatchNormalization(scale=False)(convB1_31)
    convB1_31 = Activation('relu')(convB1_31)
    
    convB1_41 = Conv2D(192//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convB1_41 = BatchNormalization(scale=False)(convB1_41)
    convB1_41 = Activation('relu')(convB1_41)
    
    concatB1 = Concatenate()([convB1_15, convB1_23, convB1_31, convB1_41])
    
    return concatB1

def resizeB(ratio, input_):
    branch3x3 = Conv2D(192//ratio,(1, 1), strides = (1,1), use_bias = False, padding = 'same', kernel_regularizer=l2(0.02))(input_)
    branch3x3 = BatchNormalization(scale=False)(branch3x3)
    branch3x3 = Activation('relu')(branch3x3)
    branch3x3 = Conv2D(320//ratio, (3, 3), strides=(2, 2), use_bias = False, padding='valid', kernel_regularizer=l2(0.02))(branch3x3)
    branch3x3 = BatchNormalization(scale=False)(branch3x3)
    branch3x3 = Activation('relu')(branch3x3)

    branch7x7x3 = Conv2D(192//ratio, (1, 1), strides = (1,1), use_bias = False, padding = 'same', kernel_regularizer=l2(0.02))(input_)
    branch7x7x3 = BatchNormalization(scale=False)(branch7x7x3)
    branch7x7x3 = Activation('relu')(branch7x7x3)
    branch7x7x3 = Conv2D(192//ratio, (1, 7), strides = (1,1), use_bias = False, padding = 'same', kernel_regularizer=l2(0.02))(branch7x7x3)
    branch7x7x3 = BatchNormalization(scale=False)(branch7x7x3)
    branch7x7x3 = Activation('relu')(branch7x7x3)
    branch7x7x3 = Conv2D(192//ratio, (7, 1), strides = (1,1), use_bias = False, padding = 'same', kernel_regularizer=l2(0.02))(branch7x7x3)
    branch7x7x3 = BatchNormalization(scale=False)(branch7x7x3)
    branch7x7x3 = Activation('relu')(branch7x7x3)
    branch7x7x3 = Conv2D(192//ratio, (3, 3), strides = (2, 2), use_bias = False, padding='valid', kernel_regularizer=l2(0.02))(branch7x7x3)
    branch7x7x3 = BatchNormalization(scale=False)(branch7x7x3)
    branch7x7x3 = Activation('relu')(branch7x7x3)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(input_)
    x = Concatenate()([branch3x3, branch7x7x3, branch_pool])
    
    return x

def moduleC(ratio, input_):
    convC1_11 = Conv2D(448//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convC1_11 = BatchNormalization(scale=False)(convC1_11)
    convC1_11 = Activation('relu')(convC1_11)
    convC1_12 = Conv2D(384//ratio, (3,3), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convC1_11)
    convC1_12 = BatchNormalization(scale=False)(convC1_12)
    convC1_12 = Activation('relu')(convC1_12)
    convC1_13 = Conv2D(384//ratio, (1,3), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convC1_12)
    convC1_13 = BatchNormalization(scale=False)(convC1_13)
    convC1_13 = Activation('relu')(convC1_13)
    convC1_14 = Conv2D(384//ratio, (3,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convC1_12)
    convC1_14 = BatchNormalization(scale=False)(convC1_14)
    convC1_14 = Activation('relu')(convC1_14)
    concatC1_1 = Concatenate()([convC1_13, convC1_14])
    
    convC1_21 = Conv2D(384//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convC1_21 = BatchNormalization(scale=False)(convC1_21)
    convC1_21 = Activation('relu')(convC1_21)
    convC1_22 = Conv2D(384//ratio, (1,3), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convC1_21)
    convC1_22 = BatchNormalization(scale=False)(convC1_22)
    convC1_22 = Activation('relu')(convC1_22)
    convC1_23 = Conv2D(384//ratio, (3,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(convC1_21)
    convC1_23 = BatchNormalization(scale=False)(convC1_23)
    convC1_23 = Activation('relu')(convC1_23)
    concatC1_2 = Concatenate()([convC1_22, convC1_23])
    
    poolC1 = AveragePooling2D(pool_size=(3, 3), strides = (1,1), padding='same')(input_)
    convC1_31 = Conv2D(192//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(poolC1)
    convC1_31 = BatchNormalization(scale=False)(convC1_31)
    convC1_3 = Activation('relu')(convC1_31)
    
    convC1_41 = Conv2D(320//ratio, (1,1), use_bias = False, padding='same', kernel_regularizer=l2(0.02))(input_)
    convC1_41 = BatchNormalization(scale=False)(convC1_41)
    convC1_4 = Activation('relu')(convC1_41)
    
    concatC1 = Concatenate()([concatC1_1, concatC1_2, convC1_3, convC1_4])
    
    return concatC1

def inceptionv3(img_input, ratio = 1, num_A = 3, num_B = 4, num_C = 2, num_class = 1000, dropout = 0.5):
    conv1_1 = Conv2D(32//ratio, (3,3), strides = (2,2), use_bias = False, padding='valid', name='conv1_1', kernel_regularizer=l2(0.02))(img_input)    
    conv1_1 = BatchNormalization(scale = False)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(32//ratio, (3,3), strides = (1,1), use_bias = False, padding='valid', name='conv1_2', kernel_regularizer=l2(0.02))(conv1_1)
    conv1_2 = BatchNormalization(scale = False)(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)
    conv1_3 = Conv2D(64//ratio, (3,3), strides = (1,1), use_bias = False, padding='same', name='conv1_3', kernel_regularizer=l2(0.02))(conv1_2)
    conv1_3 = BatchNormalization(scale = False)(conv1_3)
    conv1_3 = Activation('relu')(conv1_3)
    pool1 = MaxPooling2D((3,3), strides = (2,2), name = 'pool1')(conv1_3)
    conv1_4 = Conv2D(80//ratio, (1,1), strides = (1,1), use_bias = False, padding='valid', name='conv1_4', kernel_regularizer=l2(0.02))(pool1)
    conv1_4 = BatchNormalization(scale = False)(conv1_4)
    conv1_4 = Activation('relu')(conv1_4)
    conv1_5 = Conv2D(192//ratio, (3,3), strides = (1,1), use_bias = False, padding='valid', name='conv1_5', kernel_regularizer=l2(0.02))(conv1_4)
    conv1_5 = BatchNormalization(scale = False)(conv1_5)
    conv1_5 = Activation('relu')(conv1_5)
    x = MaxPooling2D((3,3), strides = (2,2), name = 'pool2')(conv1_5)
    
    for i in range (num_A):
        x = moduleA(ratio, x)
        
    x = resizeA(ratio, x)
    
    x = moduleB(ratio, x, num = 128//ratio)    
    for i in range(num_B - 2):
        x = moduleB(ratio, x, num = 160//ratio)    
    x = moduleB(ratio, x, num = 192//ratio)
    
    x = resizeB(ratio, x)
    
    for i in range(num_C):
        x = moduleC(ratio, x)
    
    #End
    pool = GlobalAveragePooling2D()(x)
    dropout1 = Dropout(dropout)(pool)
    patch_output = Dense(num_class, activation = 'softmax', name='patch_output')(dropout1)
    
    Classifier = Model(img_input, patch_output)
    return Classifier
