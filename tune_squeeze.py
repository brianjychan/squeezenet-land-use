"""
Using transfer learning with SqueezeNet code trained on imagenet
"""
import datetime
import sys
import os
import code_squeezenet
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras import utils
import numpy as np

# DEV = False
# argvs = sys.argv
# argc = len(argvs)

# if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
#   DEV = True

# if DEV:
#   epochs = 2
# else:
#   epochs = 15

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

model = code_squeezenet.SqueezeNet_Tune(include_top = False)
print('Model loaded.')
print("All layers: {}".format(model.layers))
print("num layers: {}".format(len(model.layers)))
print("sqz model has: {}".format(model.summary()))

# Up to but not including the last fire block
for layer in model.layers[:33]:
    layer.trainable = False


print("new model: {}".format(model.summary()))

model.compile(optimizer= optimizers.SGD(lr=1e-6, momentum=.9), metrics=['accuracy'], loss='categorical_crossentropy')


#now we have a model that has weights pre instantiated and is ready to train

###############################
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )


valid_generator = valid_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )

model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples //batch_size,
    epochs = epochs,
    validation_data = valid_generator,
    validation_steps= nb_validation_samples // batch_size,
    shuffle=False
)
