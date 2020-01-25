#!/usr/bin/env python
# coding: utf-8

# # Retraining Inception V3
# 
# In this notebook, I will go over steps to retrain Inception V3 for the skin cancer dataset.
import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Load in the Dataset
X_train = np.load("/floyd/input/skin_cancer_192_256/256_192_train.npy")
y_train = np.load("/floyd/input/skin_cancer_192_256/train_labels.npy")
X_val = np.load("/floyd/input/skin_cancer_192_256/256_192_val.npy")
y_val = np.load("/floyd/input/skin_cancer_192_256/val_labels.npy")
X_train.shape, X_val.shape
y_train.shape, y_val.shape
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_train.shape, y_val.shape

# ## Load in Pretrained Inception Model
pre_trained_model = InceptionV3(input_shape=(192, 256, 3), include_top=False, weights="imagenet")
for layer in pre_trained_model.layers:
    print(layer.name)
    layer.trainable = False  
print(len(pre_trained_model.layers))

last_layer = pre_trained_model.get_layer('mixed10')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# ## Define the Model
# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.7
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(7, activation='softmax')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()


# ## Training
# 
# Here we retrain using the whole model instead of performing transfer learning. The reason for this is the issue with batch-normalization layer implemented in Keras. During training the network will always use the mini-batch statistics either the BN layer is frozen or not; also during inference it will use the previously learned statistics of the frozen BN layers. As a result, if we fine-tune the top layers, their weights will be adjusted to the mean/variance of the new dataset. Nevertheless, during inference they will receive data which are scaled differently because the mean/variance of the original dataset will be used. Consequently, performing transfer-learning with InceptionV3 will result in very bad validation accuracy. For solution to this issue, have a look at: https://github.com/keras-team/keras/pull/9965 and https://github.com/keras-team/keras/issues/9214. For now, let's just retrain the whole model with very small learning_rate = 0.0001 and large momentum and a learning_rate_reduction function that halves the learning whenever the validation accuracy doesn't change for a 3 consecutive epochs. We will train for only 20 epochs so that the weights of the original pretrained model won't change too much and overfit the train data.

# ### Feature Extraction
# 
# Before we even retrain our model, it's better that we freeze all the layers in InceptionV3 and just train our top fully-connected and classification layers so that the weights for these layers won't be too random. The intuition for this is that if we didn't perform feature-extraction, then the gradient would be too large and would change the pretrained weights too much.
train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')
train_datagen.fit(X_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(X_val)

batch_size = 64
epochs = 3
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size), 
                              validation_steps=(X_val.shape[0] // batch_size))

# ### Retraining
# 
# Now, we are retraining the whole models. The goal is to just tune the weights a bit for our dataset and avoid changing the pretrained weights too much!
for layer in pre_trained_model.layers:
    layer.trainable = True


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)
model.summary()

batch_size = 64
epochs = 20
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size),
                              callbacks=[learning_rate_reduction])

loss_val, acc_val = model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))


# Even though this model overfits the training data, we also observe some significant improvement with the validation accuracy. Our final validation accuracy is 86.9%, a nearly 10% improvement from the baseline model, which justifies some more training time. We can also see that this model is extremely "sensitive", a small change in learning rate can can change the model by a whole lot. Future work on choosing learning rate as well as monitor learning rate is needed for further improvement. 

# ## Testing
# Let's load in the intact test set and test our model
X_test = np.load("/floyd/input/skin_cancer_192_256/256_192_test.npy")
y_test = np.load("/floyd/input/skin_cancer_192_256/test_labels.npy")
y_test = to_categorical(y_test)
loss_test, acc_test = model.evaluate(X_test, y_test, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))

# Achieving test accuracy of 86.8% after 20 training epochs is a good result! This experiment proved that the architecture and the weights of Inception trained on ImageNet help learning for a complete different domain dataset. 
model.save("InceptionV3.h5")

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, label = "training")
plt.plot(epochs, val_acc, label = "validation")
plt.legend(loc="upper left")
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, label = "training")
plt.plot(epochs, val_loss, label = "validation")
plt.legend(loc="upper right")
plt.title('Training and validation loss')
