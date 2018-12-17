"""
Using transfer learning with SqueezeNet code trained on imagenet
"""
import datetime
import sys
import os
import code_squeezenet
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
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
img_width, img_height = 150, 150

name = 'vgg'
nb_train_samples = 25200
nb_validation_samples = 3150
batch_size = 32
classes_num = 45
epochs = 25
lr = .001


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])


    generator = datagen.flow_from_directory(
        train_data_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size + 1)
    with open('vgg_bottleneck_features_train.npy', 'wb') as features_train_file: 
        np.save(features_train_file, bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size + 1)
    with open('vgg_bottleneck_features_validation.npy', 'wb') as features_validation_file:
        np.save(features_validation_file, bottleneck_features_validation)

def train_top_model():
    with open('vgg_bottleneck_features_train.npy', 'rb') as train_data_file:
        train_data = np.load(train_data_file)
    
    train_labels = np.arange(45)
    train_labels = np.repeat(train_labels, 560)
    
    train_labels = utils.to_categorical(train_labels, num_classes=45)


    with open('vgg_bottleneck_features_validation.npy', 'rb') as validation_data_file:
        validation_data = np.load(validation_data_file)
    
    validation_labels = np.arange(45)
    validation_labels = np.repeat(validation_labels, 70)

    validation_labels = utils.to_categorical(validation_labels, num_classes=45)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    
    history_callback = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    loss_history = history_callback.history['loss']
    numpy_loss_history = np.array(loss_history)

    np.savetxt("./losses/loss_history-{}-{}.txt".format(name, datetime.datetime.now()), numpy_loss_history, delimiter=",")

    model.save('./models/{}-{}.h5'.format(name, datetime.datetime.now()))
    model.save_weights('./models/{}_weights-{}.h5'.format(name, datetime.datetime.now()))


save_bottleneck_features()
train_top_model()







































# # train topew
# def train_top_model():
#     with open('vgg_bottleneck_features_train.npy', 'rb') as train_data_file:
#         train_data = np.load(train_data_file)
    
#     train_labels = np.arange(45)
#     train_labels = np.repeat(train_labels, 560)
    
#     train_labels = utils.to_categorical(train_labels, num_classes=45)


#     with open('vgg_bottleneck_features_validation.npy', 'rb') as validation_data_file:
#         validation_data = np.load(validation_data_file)
    
#     validation_labels = np.arange(45)
#     validation_labels = np.repeat(validation_labels, 70)

#     validation_labels = utils.to_categorical(validation_labels, num_classes=45)

#     model = Sequential()
#     print('train_data shape: {}'.format(train_data.shape))
#     print('train_labels shape: {}'.format(train_labels.shape))

#     print('val_data shape: {}'.format(validation_data.shape))
#     print('val_labels shape: {}'.format(validation_labels.shape))
#     model.add(Dense(1024, activation='relu', input_shape = train_data.shape[1:]))
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(classes_num, activation='softmax'))

#     # sgd = optimizers.SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)

#     model.compile(optimizer=optimizers.RMSprop(lr=lr),
#                   loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#     model.compile(optimizer=optimizers.RMSprop(lr=lr),
#                   loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#     history_callback = model.fit(train_data, train_labels,
#               epochs=epochs,
#               batch_size=batch_size,
#               validation_data=(validation_data, validation_labels))
#     loss_history = history_callback.history['loss']
#     numpy_loss_history = np.array(loss_history)

#     np.savetxt("./losses/loss_history-{}-{}.txt".format(name, datetime.datetime.now()), numpy_loss_history, delimiter=",")

#     model.save('./models/{}-{}.h5'.format(name, datetime.datetime.now()))
#     model.save_weights('./models/{}_weights-{}.h5'.format(name, datetime.datetime.now()))


# train_top_model()