"""
Adapted from "CNN Image Classifier" from Github user tatsuyah at https://github.com/tatsuyah/CNN-Image-Classifier.

Pairs with predict.py
Basic but inflexible code for creating a CNN model.

"""
import datetime
import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks

train_data_path = './data/train'
validation_data_path = './data/validation'

"""
Parameters
"""
img_width, img_height = 256,256
batch_size = 64
steps_per_epoch = 394
# samples_per_epoch = 29740
validation_steps = 27
nb_filters1 = 256
nb_filters2 = 256
nb_filters4 = 384
nb_filters5 = 384
nb_filters6 = 256
conv1_size = 5
conv2_size = 3
conv3_size = 3
pool_size = 2
classes_num = 45
lr = 0.001
epochs = 15

name = '3lay'

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, padding ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))

model.add(Convolution2D(nb_filters2, conv2_size, padding ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format='channels_first'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
              

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = valid_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    steps_per_epoch= steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
  
model.save('./models/{}-{}.h5'.format(name, datetime.datetime.now()))
model.save_weights('./models/{}weights{}.h5'.format(name, datetime.datetime.now()))
