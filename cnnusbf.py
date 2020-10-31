#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to implement the network in Google Colab for access to hardware 
accelerator i.e. GPUs. With GPUs, the training is accelerated manifold.

@author: rpm1412
"""

#%% Cell 1: Import libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Import libraries for the Neural Network Section
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Reshape, Multiply, Lambda, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from google.colab import drive
from scipy.signal import hilbert

#%% Cell 2: Setting variables

start_epoch = 0
epochs = 5
stop_epoch = start_epoch + epochs

train_scans = 600 #Number of scanning data frames

# Image/Time of Flight data dimensions. Modify according to need 
rows = 374
cols = 128
channels = 128

'''
# It is preferable to name the scan data as follows:
# Time of flight data : Dimension : Rows x Cols x Channels
# MVDR Data prior to hilbert transform & log compression : Dimension : Rows x Cols

#Name them with numbers as indexes. An example is. tofc_1,tofc_2 and mvdr_1,mvdr_2
#Naming them with numerical index gives an advantage to pick them for training.
'''

#%% Cell 3: Making the network

class Antirectifier(layers.Layer):
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # make sure it is a 3D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=-1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=-1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=-1)

#CNN Model

inputs = Input(shape=(rows,cols,channels)) #Input layer
# Normalizing inputs using channel as axis
inputs_norm = Lambda(lambda x : K.l2_normalize(x,axis=-1))(inputs)

output_1 = Conv2D(32,(3,3), padding='same',kernel_initializer='glorot_normal')(inputs_norm)
act_1 = Antirectifier()(output_1)
B1 = BatchNormalization()(act_1)

output_2 = Conv2D(32,(3,3), padding='same',kernel_initializer='glorot_normal')(B1)
act_2 = Antirectifier()(output_2)
B2 = BatchNormalization()(act_2)

output_3 = Conv2D(64,(3,3), padding='same',kernel_initializer='glorot_normal')(B2)
act_3 = Antirectifier()(output_3)
B3 = BatchNormalization()(act_3)

output_4 = Conv2D(64,(3,3), padding='same',kernel_initializer='glorot_normal')(B3)
act_4 = Antirectifier()(output_4)
B4 = BatchNormalization()(act_4)

# Adaptive weights
output_5 = Conv2D(channels,(3,3), activation = "softmax", padding='same')(B4)

#Beamforming
beamform = Multiply()([inputs,output_5])
beamform_sum = Lambda(lambda x: K.sum(x, axis=-1))(beamform)

output_fig = Reshape((rows,cols))(beamform_sum)
model = Model(inputs=inputs, outputs=output_fig)

# Print a model summary
model.summary()

#%% Cell 4: Compiling the model

def msle(y_true, y_pred):
  y_true = K.cast(y_true, y_pred.dtype)
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  first_log = K.log(K.clip(K.abs(y_pred), K.epsilon(), None) )
  second_log = K.log(K.clip(K.abs(y_true), K.epsilon(), None) )
  return K.mean(K.square(first_log - second_log), axis=-1)

def mse(y_true, y_pred):
  y_true = K.cast(y_true, y_pred.dtype)
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  return K.mean(K.square(y_true - y_pred), axis=-1)

learning_rate = 1e-4
adam_lr = keras.optimizers.Adam(lr = learning_rate)
model.compile(optimizer = adam_lr, loss= mse)

#%% Cell 5: Train the model

# Loading Dataset and Training
drive.mount('/content/gdrive')

'''
# Best to save data with the number 1 to 600 as it will be easy to pick them
#to train as mentioned above in cell 2.
'''

# Produces indexed for picking data to train
list_train = np.add(1,np.arange(train_scans))

X_final = np.zeros((30,rows,cols,channels)) # Fixing batch size to be 30
y_final = np.zeros((30,rows,cols)) # Fixing batch size to be 30

for epoch in range(start_epoch, stop_epoch): # Run through all epochs
  np.random.shuffle(list_train) # Randomize the entry of data to train
  for batch in range(0,600,30):
    # To load one batch at a time to prevent memory issues
    for scan in range(30):
      j=list_train[batch+scan]
      # Load your data here one by one by using j as index. 
      # As mentioned in cell 2, eg. 'tofc_'+str(j) or 'mvdr_'+str(j)

      #X= load_ToFC_data_using_j_as_index #dimensions:(rows,cols,channels)
      #y= load_mvdr_data_using_j_as_index #dimensions:(rows,cols)

      X_final[scan,:,:,:] = X #storing loaded data as a batch
      y_final[scan,:,:] = y   #storing loaded data as a batch

    model.fit(X_final, y_final, batch_size = 30, epochs=epoch+1, initial_epoch=epoch, shuffle=True, verbose = 1)  # starts training
    
  file_name_w = '/content/gdrive/My Drive/<your_path>/model.h5' #Enter your path
  model.save(file_name_w)

#%% Cell 6: Loading model from Google drive for prediction during validation

#Pre loading saved model
drive.mount('/content/gdrive')

#Please enter your filepath to load saved model file.
file_name = '/content/gdrive/My Drive/<your_path>/model.h5'
model.load_weights(file_name)


#%% Cell 7: Predicting on validation data
  
#X = load_validation_time_of_flight_data #dimensions:(rows,cols,channels)
Xf[0,:,:,:] = X # Converting to suit entry into the network
#y = load_validation_mvdr_data #dimensions:(rows,cols)
  
test = model.predict(Xf, batch_size = 1, verbose = 1)  
test_image = np.reshape(test[0],(rows,cols))

validation_image = y

#Hilbert transform for B mode imaging and Log compression 
test_image = 20*np.log10(np.abs(hilbert(test_image))) 
MVDR_image = 20*np.log10(np.abs(hilbert(validation_image)))

#Plot the images
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,10))
ax1.set_title("CNN")
im1 = ax1.imshow(test_image, cmap='gray')
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax = cax1)

ax2.set_title("MVDR")
im2 = ax2.imshow(MVDR_image, cmap='gray')
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax= cax2)
