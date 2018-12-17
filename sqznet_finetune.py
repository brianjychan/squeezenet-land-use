"""
Driver code for the fine-tuning of a SqueezeNet model. 

Corresponds with squeeze_train.train_top_model_functional() and code_squeezenet.SqueezeNet_Tune() 
to create the required weights for fine tuning. You cannot begin training the top model
with randomized final layer weights; they must already be trained as well. Consult
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
for more info.

Can be used as driver code for transfer learning with the models from sqznet_models
Instantiates a SqueezeNet model that also has its final layers trained by using code_squeezenet.SqueezeNet_Tune().
"""
import datetime
import sys
import os
import sqznet_models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras import utils
import numpy as np

train_data_path = './data/train'
validation_data_path = './data/validation'

"""
Parameters
"""
img_width, img_height = 64, 64

name = 'squeeze'
nb_train_samples = 25200
nb_validation_samples = 3150
batch_size = 64
classes_num = 45
epochs = 25
lr = .001

model = sqznet_models.SqueezeNet_Tune(include_top = False)
print('Model loaded.')
print(model.summary())

# Up to but not including the last fire block of convolutional layers
for layer in model.layers[:33]:
    layer.trainable = False

model.compile(optimizer= optimizers.SGD(lr=1e-6, momentum=.9), metrics=['accuracy'], loss='categorical_crossentropy')


# Set up training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )

valid_datagen = ImageDataGenerator(rescale=1. / 255)
valid_generator = valid_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )

# Train the model
model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples //batch_size,
    epochs = epochs,
    validation_data = valid_generator,
    validation_steps= nb_validation_samples // batch_size,
    shuffle=False
)
