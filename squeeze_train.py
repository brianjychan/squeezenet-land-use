"""
Using transfer learning with SqueezeNet code trained on imagenet
"""
import datetime
import sys
import os
import code_squeezenet
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras import utils
import numpy as np
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation, Dense


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

model = code_squeezenet.SqueezeNet(include_top = False)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )

bottleneck_features_train = model.predict_generator(generator, 25200 // batch_size + 1)
with open('bottleneck_features_train.npy', 'wb') as features_train_file: 
    np.save(features_train_file, bottleneck_features_train)

valid_generator = train_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )

bottleneck_features_validation = model.predict_generator(valid_generator, 3150 // batch_size + 1)
with open('bottleneck_features_validation.npy', 'wb') as features_validation_file:
    np.save(features_validation_file, bottleneck_features_validation)


# train topew
def train_top_model():
    with open('bottleneck_features_train.npy', 'rb') as train_data_file:
        train_data = np.load(train_data_file)
    
    train_labels = np.arange(45)
    train_labels = np.repeat(train_labels, 560)
    
    train_labels = utils.to_categorical(train_labels, num_classes=45)


    with open('bottleneck_features_validation.npy', 'rb') as validation_data_file:
        validation_data = np.load(validation_data_file)
    
    validation_labels = np.arange(45)
    validation_labels = np.repeat(validation_labels, 70)

    validation_labels = utils.to_categorical(validation_labels, num_classes=45)

    model = Sequential()
    print('train_data shape: {}'.format(train_data.shape))
    print('train_labels shape: {}'.format(train_labels.shape))

    print('val_data shape: {}'.format(validation_data.shape))
    print('val_labels shape: {}'.format(validation_labels.shape))
    model.add(Dense(512, activation='relu', input_shape = train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))



    model.compile(optimizer=optimizers.RMSprop(lr=lr),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.compile(optimizer=optimizers.RMSprop(lr=lr),
    #               loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    history_callback = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    loss_history = history_callback.history['loss']
    numpy_loss_history = np.array(loss_history)

    np.savetxt("./losses/loss_history-{}-{}.txt".format(name, datetime.datetime.now()), numpy_loss_history, delimiter=",")

    model.save('./models/{}-{}.h5'.format(name, datetime.datetime.now()))
    model.save_weights('./models/{}_weights-{}.h5'.format(name, datetime.datetime.now()))

# for functional
def train_top_model_functional():
    name = "top_model_functional"
    with open('bottleneck_features_train.npy', 'rb') as train_data_file:
        train_data = np.load(train_data_file)
    
    train_labels = np.arange(45)
    train_labels = np.repeat(train_labels, 560)
    
    train_labels = utils.to_categorical(train_labels, num_classes=45)


    with open('bottleneck_features_validation.npy', 'rb') as validation_data_file:
        validation_data = np.load(validation_data_file)
    
    validation_labels = np.arange(45)
    validation_labels = np.repeat(validation_labels, 70)

    validation_labels = utils.to_categorical(validation_labels, num_classes=45)
    
 
    inputs = Input(shape=(512,))
    x = Dense(512, activation='relu', name = 'top_dense_1')(inputs)
    x = Dense(512, activation='relu', name = 'top_dense_2')(x)
    x = Dropout(.5)(x)
    x = Dense(45, activation = 'softmax', name = 'top_final')(x)

    model = Model(inputs, x, name="topmodel")
    model.compile(optimizer=optimizers.RMSprop(lr=lr),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print("summary: {}".format(model.summary()))

    #when we run this, it will give us weights for those last two layers

    history_callback = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    loss_history = history_callback.history['loss']
    numpy_loss_history = np.array(loss_history)

    np.savetxt("./losses/loss_history-{}-{}.txt".format(name, datetime.datetime.now()), numpy_loss_history, delimiter=",")

    model.save('./models/{}-{}.h5'.format(name, datetime.datetime.now()))
    model.save_weights('./models/{}_weights-{}.h5'.format(name, datetime.datetime.now()))

# train_top_model()
train_top_model_functional()