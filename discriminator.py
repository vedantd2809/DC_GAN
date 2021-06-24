# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:18:33 2021

@author: vedant
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Reshape,Dense,Conv2DTranspose,BatchNormalization,LeakyReLU,ReLU,Add,Activation,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
#from generator import *
#from model import *

def discriminator_mod(d):
    i = Input(shape=(d))
    x = Dense(2048,activation=LeakyReLU())(i)
    x = Dense(1024,activation=LeakyReLU())(x)
    x = Dense(512,activation=LeakyReLU())(x)
    x = Dense(256, activation=LeakyReLU())(x)
    x = Dense(128, activation=LeakyReLU())(x)
    x = Dense(1,activation='sigmoid')(x)
    
    model = Model(i, x)
    
    return model