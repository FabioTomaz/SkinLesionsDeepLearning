#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# ## Load in the Validation and Test Set
X_val = np.load("/Users/Hoang/Machine_Learning/skin_cancer/skin_cancer_192_256/256_192_val.npy")
X_test = np.load("/Users/Hoang/Machine_Learning/skin_cancer/skin_cancer_192_256/256_192_test.npy")
y_val = np.load("/Users/Hoang/Machine_Learning/skin_cancer/skin_cancer_192_256/val_labels.npy")
y_test = np.load("/Users/Hoang/Machine_Learning/skin_cancer/skin_cancer_192_256/test_labels.npy")
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
X_val.shape, X_test.shape, y_val.shape, y_test.shape

input_shape = X_val[0,:,:,:].shape
model_input = Input(shape=input_shape)

# ## Define InceptionV3
inception = InceptionV3(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
for layer in inception.layers:
    layer.trainable = True

inception_last_layer = inception.get_layer('mixed10')
print('last layer output shape:', inception_last_layer.output_shape)
inception_last_output = inception_last_layer.output

# Flatten the output layer to 1 dimension
x_inception = layers.GlobalMaxPooling2D()(inception_last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x_inception = layers.Dense(512, activation='relu')(x_inception)
# Add a dropout rate of 0.7
x_inception = layers.Dropout(0.5)(x_inception)
# Add a final sigmoid layer for classification
x_inception = layers.Dense(7, activation='softmax')(x_inception)

# Configure and compile the model
inception_model = Model(model_input, x_inception)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
inception_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
inception_model.load_weights("InceptionV3full.h5")
inception_model.summary()

# ## Define DenseNet 201 
denseNet = DenseNet201(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
for layer in denseNet.layers:
    layer.trainable = True


denseNet_last_layer = denseNet.get_layer('relu')
print('last layer output shape:', denseNet_last_layer.output_shape)
denseNet_last_output = denseNet_last_layer.output

# Flatten the output layer to 1 dimension
x_denseNet = layers.GlobalMaxPooling2D()(denseNet_last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x_denseNet = layers.Dense(512, activation='relu')(x_denseNet)
# Add a dropout rate of 0.7
x_denseNet = layers.Dropout(0.5)(x_denseNet)
# Add a final sigmoid layer for classification
x_denseNet = layers.Dense(7, activation='softmax')(x_denseNet)

# Configure and compile the model
denseNet_model = Model(model_input, x_denseNet)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
denseNet_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
denseNet_model.load_weights("DenseNetFull.h5")
denseNet_model.summary()

# ## Define Ensemble Model
def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = layers.Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model

ensemble_model = ensemble([denseNet_model, inception_model], model_input)
ensemble_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# ## Testing
loss_val, acc_val = ensemble_model.evaluate(X_val, y_val, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))

loss_test, acc_test = ensemble_model.evaluate(X_test, y_test, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))
