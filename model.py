# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:19:08 2021

@author: vedant
"""

import numpy as np
import cv2
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Conv2D,Conv2DTranspose,Reshape,BatchNormalization,Add,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from generator import *
from discriminator import *

path = "C:/Users/vedant/ML&AI/Vedant/res_pap/1/images/images/"
list_ = os.listdir(path)

h = 256
w = 256
c = 3
d = h*w

img = []
for i in list_:
    a = cv2.imread(path+i)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    a = cv2.resize(a,(h,w))
    a = a.reshape(h, w,3)
    img.append(a)
    
plt.imshow(img[1])
img = np.array(img)
img = img/255.0*2-1

discriminator = discriminator_mod(d)
discriminator.compile(loss='categorical_crossentropy',optimizer=Adam(0.0002,0.5),metrics=['accuracy'])
generator = generator_mod(latent_dim)

discriminator.summart()
generator.summary()

z = Input(shape=(latent_dim,))
z.shape

gen = generator(z)
gen

discriminator.trainable = False
fake_pred = discriminator(gen)

combined = Model(z,fake_pred)
combined.compile(loss='binary_crossentropy',optimizer = Adam(0.0002,0.6))

batch_size = 128
epochs = 10000
sample = 500

zeros = np.zeros(batch_size)
ones = np.ones(batch_size)

d_losses = []
g_losses = []


if not os.path.exists('gan_img'):
    os.makedirs('gan_img')
    

def sample_images(epoch):
    rows,cols = 5,5
    noise = np.random.randn(rows*cols,latent_dim)
    fimg = generator.predict(noise)
    
    fimg = 0.5*fimg+0.5
    
    fig,axs = plt.subplots(rows,cols)
    idx = 0
    
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(fimg[idx].reshape(h,w),cmap='bone')
            axs[i,j].axis('off')
            idx+=1
    fig.savefig("gan_img/%d.png" % epoch)
    plt.close()
    

for epoch in range(epochs):
    idx = np.random.randn(0,img.shape[0],batch_size)
    real_img = img[idx]
    
    noise = np.random.randn(batch_size,latent_dim)
    fake_img = generator.predict(noise)
    
    
    
    plt.imshow(fake_imgs)
    
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)

    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_acc  = 0.5 * (d_acc_real + d_acc_fake)
    
    noise = np.random.randn(batch_size, latent_dim)
    g_loss = combined_model.train_on_batch(noise, ones)
    
    
    noise = np.random.randn(batch_size, latent_dim)
    g_loss = combined_model.train_on_batch(noise, ones)
    
    
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    
    if epoch % 100 == 0:
      print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, \
        d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
    
    if epoch % sample_period == 0:
      sample_images(epoch)