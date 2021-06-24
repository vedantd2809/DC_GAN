# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:18:46 2021

@author: vedant
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Reshape,Dense,BatchNormalization,Flatten,LeakyReLU,Add,Activation,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
#from discriminator import *
#from model import *


latent_dim = 100
#d = h*w

def generator_mod(latent_dim):
    i = Input(shape=(latent_dim,))
    x = Dense(16*16*64,input_shape=(100,))(i)
    x = BatchNormalization()(x)
    x = LeakyReLU()
    
    x = Reshape((8,8,256))(x)
    
    x = Conv2DTranspose(256, (2,2), strides = 1,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(256, (3,3), strides = 2,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, (4,4), strides = 2,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(64, (4,4), strides = 2,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(32, (4,4), strides = 2,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(3, (5,5), strides=2,padding='same',activation='tanh')(x)
    
    model = Model(i,x)
    return model