#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 20:56:47 2018

@author: valaxw
"""
from math import ceil
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Dense, MaxPooling2D, Flatten, Dropout, Activation, Add, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

def VGG16(img_input, ratio, num_classes, dropout = 0.5):
    conv1_1 = Conv2D(64//ratio, (3,3), activation = 'relu', padding = 'same')(img_input)
    conv1_2 = Conv2D(64//ratio, (3,3), activation = 'relu', padding = 'same')(conv1_1)
    pool1 = MaxPooling2D((3,3), strides=(2,2))(conv1_2)
    
    conv2_1 = Conv2D(128//ratio, (3,3), activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128//ratio, (3,3), activation = 'relu', padding = 'same')(conv2_1)
    pool2 = MaxPooling2D((3,3), strides = (2,2))(conv2_2)
    
    conv3_1 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_1)
    conv3_3 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_2)
    pool3 = MaxPooling2D((3,3), strides = (2,2))(conv3_3)
    
    conv4_1 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_1)
    conv4_3 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_2)
    pool4 = MaxPooling2D((3,3), strides = (2,2))(conv4_3)
    
    conv5_1 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(pool4)
    conv5_2 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_1)
    conv5_3 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_2)
    pool5 = MaxPooling2D((3,3), strides = (2,2))(conv5_3)
    pool5 = Flatten()(pool5)

    fc6 = Dense(4096//ratio, activation = 'relu')(pool5)
    dropout1 = Dropout(dropout)(fc6)
    fc7 = Dense(4096//ratio, activation = 'relu')(dropout1)
    dropout2 = Dropout(dropout)(fc7)
    fc8 = Dense(num_classes, activation = 'softmax')(dropout2)
    model = Model(img_input, fc8)
    return model

def VGG19(img_input, ratio, num_classes, dropout = 0.5):
    conv1_1 = Conv2D(64//ratio, (3,3), activation = 'relu', padding = 'same')(img_input)
    conv1_2 = Conv2D(64//ratio, (3,3), activation = 'relu', padding = 'same')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1_2)
    
    conv2_1 = Conv2D(128//ratio, (3,3), activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128//ratio, (3,3), activation = 'relu', padding = 'same')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides = (2,2))(conv2_2)
    
    conv3_1 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_1)
    conv3_3 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_2)
    conv3_4 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_3)
    pool3 = MaxPooling2D((2,2), strides = (2,2))(conv3_4)
    
    conv4_1 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_1)
    conv4_3 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_2)
    conv4_4 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_3)
    pool4 = MaxPooling2D((2,2), strides = (2,2))(conv4_4)
    
    conv5_1 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(pool4)
    conv5_2 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_1)
    conv5_3 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_2)
    conv5_4 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_3)
    pool5 = MaxPooling2D((2,2), strides = (2,2))(conv5_4)
    pool5 = Flatten()(pool5)
    
    fc6 = Dense(4096//ratio, activation = 'relu')(pool5)
    dropout1 = Dropout(dropout)(fc6)
    fc7 = Dense(4096//ratio, activation = 'relu')(dropout1)
    dropout2 = Dropout(dropout)(fc7)
    fc8 = Dense(num_classes, activation = 'softmax')(dropout2)
    
    model = Model(img_input, fc8)
    return model

def VGG_modified(img_input, ratio, num_classes, dropout = 0.7):
    conv1_1 = Conv2D(64//ratio, (3,3), activation = 'relu', padding = 'same')(img_input)
    conv1_2 = Conv2D(64//ratio, (3,3), activation = 'relu', padding = 'same')(conv1_1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1_2)
    
    conv2_1 = Conv2D(128//ratio, (3,3), activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128//ratio, (3,3), activation = 'relu', padding = 'same')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides = (2,2))(conv2_2)
    
    conv3_1 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_1)
    conv3_3 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_2)
    conv3_4 = Conv2D(256//ratio, (3,3), activation = 'relu', padding = 'same')(conv3_3)
    pool3 = MaxPooling2D((2,2), strides = (2,2))(conv3_4)
    
    conv4_1 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_1)
    conv4_3 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_2)
    conv4_4 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv4_3)
    pool4 = MaxPooling2D((2,2), strides = (2,2))(conv4_4)
    
    conv5_1 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(pool4)
    conv5_2 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_1)
    conv5_3 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_2)
    conv5_4 = Conv2D(512//ratio, (3,3), activation = 'relu', padding = 'same')(conv5_3)
    
    pool = GlobalAveragePooling2D()(conv5_4)
    fc7 = Dense(4096//ratio, activation = 'relu')(pool)
    dropout2 = Dropout(dropout)(fc7)
    fc8 = Dense(num_classes, activation = 'softmax')(dropout2)
    
    model = Model(img_input, fc8)
    return model


    