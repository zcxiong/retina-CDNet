###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import ast, operator
import matplotlib.pyplot as plt
import configparser
import time

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, SeparableConv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Reshape, core, Dropout, Activation, BatchNormalization, Flatten, Dense, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate, multiply, subtract, average, add, maximum, dot
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,History,BaseLogger,Callback,EarlyStopping, ReduceLROnPlateau
from keras.initializers import TruncatedNormal, he_normal
#from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam,Adadelta,RMSprop
from keras import losses
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
from keras.engine.topology import Layer
from keras import backend as BK
from sklearn.metrics import log_loss

init_lr = 1e-2

def get_segBlock(input, LayerName, filter, kernel_sz=(3, 3), str=(1, 1), pad='valid', dataFormat=None, dilationRate=(1, 1), act=None, useBias=True, kernelInitializer='glorot_uniform', biasInitializer='zeros', kernelRegularizer=None, biasRegularizer=None, activityRegularizer=None, kernelConstraint=None, biasConstraint=None):
    conv10 = Conv2D(filter, kernel_sz, padding="valid", activation="relu", use_bias=True, name = LayerName+"0")(input)
    do10 = Dropout(0.2, name = LayerName+"_do10")(conv10)
    conv11 = Conv2D(filter, kernel_sz, padding="same", activation="relu", use_bias=True, name = LayerName+"1")(do10)
    do11 = Dropout(0.2, name = LayerName+"_do11")(conv11)
    conca10 = concatenate([do10, do11], axis=1, name = LayerName+"_conca")
    bn1 = BatchNormalization(epsilon = 1.1e-5, name = LayerName+"_bn1")(conca10)
    #conv12 = Conv2D(filter, kernel_sz, padding="same", activation="relu", use_bias=True, name = LayerName+"2")(bn1)
    cotr12 = Conv2DTranspose(filter, (3, 3), strides=(1,1), padding='valid', activation='relu', use_bias = True, name = LayerName+"2")(bn1)
    do12 = Dropout(0.2, name = LayerName+"_do12")(cotr12)
   
    return do12

def get_CDNet(n_ch,patch_height,patch_width):
    #512
    inputs = Input((n_ch, patch_height, patch_width))
    seg1 = get_segBlock(inputs, "seg1", 32)
    #256
    pool1 = MaxPooling2D(pool_size=(2, 2), name = "pool1")(seg1)
    seg2 = get_segBlock(pool1, "seg2", 48)
    #128
    pool2 = MaxPooling2D(pool_size=(2, 2), name = "pool2")(seg2)
    seg3 = get_segBlock(pool2, "seg3", 64)
    #64
    pool3 = MaxPooling2D(pool_size=(2, 2), name = "pool3")(seg3)    
    seg4 = get_segBlock(pool3, "seg4", 80)
    #32
    pool4 = MaxPooling2D(pool_size=(2, 2), name = "pool4")(seg4)
    seg5 = get_segBlock(pool4, "seg5", 96)
    conca1 = concatenate([pool4, seg5], axis=1, name = "conca1")
    #64
    up1 = UpSampling2D(size=(2, 2))(conca1)
    seg6 = get_segBlock(up1, "seg6", 80)
    conca2 = concatenate([pool3, seg6], axis=1, name = "conca2")
    #128
    up2 = UpSampling2D(size=(2, 2))(conca2)
    seg7 = get_segBlock(up2, "seg7", 64)
    conca3 = concatenate([pool2, seg7], axis=1, name = "conca3")
    
    #256
    up3 = UpSampling2D(size=(2, 2))(conca3)
    seg8 = get_segBlock(up3, "seg8", 48)
    conca4 = concatenate([pool1, seg8], axis=1, name = "conca4")
    
    #512
    up4 = UpSampling2D(size=(2, 2))(conca4)
    seg9 = get_segBlock(up4, "seg9", 32)
    conca5 = concatenate([seg1, seg9], axis=1, name = "conca5")

    #
    conv10 = Conv2D(2, (1, 1), padding="valid", activation="relu", use_bias=True, name = "conv10")(conca5)
    RS = core.Reshape((2,patch_height*patch_width))(conv10)
    PM = core.Permute((2,1))(RS)
    ############
    ACT = core.Activation('softmax')(PM)
    model = Model(inputs=inputs, outputs=ACT)
    #model.load_weights('./'+name_experiment+'/'+name_experiment +'_best_weights.h5', by_name=True)

    sgd1 = SGD(lr=init_lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd1, loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def get_unet(n_ch,patch_height,patch_width):
    #512
    inputs = Input((n_ch, patch_height, patch_width))
    conv10 = Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True)(inputs)
    do10 = Dropout(0.2)(conv10)
    conv11 = Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True)(do10)
    do11 = Dropout(0.2)(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2))(do11)
    #256
    conv20 = Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True)(pool1)
    do20 = Dropout(0.2)(conv20)
    conv21 = Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True)(do20)
    do21 = Dropout(0.2)(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(do21)
    #128
    conv30 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(pool2)
    do30 = Dropout(0.2)(conv30)
    conv31 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(do30)
    do31 = Dropout(0.2)(conv31)
    pool3 = MaxPooling2D(pool_size=(2, 2))(do31)
    #64
    conv40 = Conv2D(512, (3, 3), padding="same", activation="relu", use_bias=True)(pool3)
    do40 = Dropout(0.2)(conv40)
    conv41 = Conv2D(512, (3, 3), padding="same", activation="relu", use_bias=True)(do40)
    do41 = Dropout(0.2)(conv41)
    pool4 = MaxPooling2D(pool_size=(2, 2))(do41)
    #32
    conv50 = Conv2D(1024, (3, 3), padding="same", activation="relu", use_bias=True)(pool4)
    do50 = Dropout(0.2)(conv50)
    conv51 = Conv2D(1024, (3, 3), padding="same", activation="relu", use_bias=True)(do50)
    do51 = Dropout(0.2)(conv51)
    up1 = concatenate([UpSampling2D(size=(2, 2))(do51), do41], axis=1)
    #64
    conv60 = Conv2D(512, (3, 3), padding="same", activation="relu", use_bias=True)(up1)
    do60 = Dropout(0.2)(conv60)
    conv61 = Conv2D(512, (3, 3), padding="same", activation="relu", use_bias=True)(do60)
    do61 = Dropout(0.2)(conv61)
    up2 = concatenate([UpSampling2D(size=(2, 2))(do61), do31], axis=1)
    #128
    conv70 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(up2)
    do70 = Dropout(0.2)(conv70)
    conv71 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(do70)
    do71 = Dropout(0.2)(conv71)
    up3 = concatenate([UpSampling2D(size=(2, 2))(do71), do21], axis=1)
    #256
    conv80 = Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True)(up3)
    do80 = Dropout(0.2)(conv80)
    conv81 = Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True)(do80)
    do81 = Dropout(0.2)(conv81)
    up4 = concatenate([UpSampling2D(size=(2, 2))(do81), do11], axis=1)
    #512
    conv90 = Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True)(up4)
    do90 = Dropout(0.2)(conv90)
    conv91 = Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True)(do90)
    do91 = Dropout(0.2)(conv91)

    conv10 = Conv2D(2, (1, 1), padding="same", activation="relu", use_bias=True)(do91)
    RS = core.Reshape((2,patch_height*patch_width))(conv10)
    PM = core.Permute((2,1))(RS)
    ############
    ACT = core.Activation('softmax')(PM)

    model = Model(input=inputs, output=ACT)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    sgd1 = SGD(lr=init_lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd1, loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def get_gnet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv5)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv6)
    #
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(32, (3, 3), padding="same", activation="relu")(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv7)
    #
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(16, (3, 3), padding="same", activation="relu")(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv8)
    #
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv9)
    #
    conv10 = Conv2D(2, (1, 1), padding="same", activation="relu")(conv9)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    #model = Model(input=inputs, output=conv10)
    model = Model(inputs=inputs, outputs=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=sgd1, loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def get_model_by_name(n_ch, patch_height, patch_width, model_name):
    switch = {"CDNet":get_CDNet, "gnet":get_gnet, "unet":get_unet}
    model = switch[model_name](n_ch, patch_height, patch_width)  #the U-net model
    model.summary()
    return model