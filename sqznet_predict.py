# need to use squeeze on test data first 
'''
Code utilized here is referenced from

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

and

https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
 

'''
import datetime
import sys
import os
import sqznet_models
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras import utils
import numpy as np
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation, Dense
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
 

img_width, img_height = 64, 64

model_path = './models/squeeze-2018-12-16 14:57:55.220107.h5'
model_weights_path = './models/squeeze_weights-2018-12-16 14:57:55.273335.h5'

test_data_path = './data/test'

"""
Parameters
"""
img_width, img_height = 64, 64


name = 'squeeze'

# Split 700 images x 45 classes  by 80:10:10 train:validation:test
nb_train_samples = 25200 # 560 * 45 classes
nb_validation_samples = 3150 #70 * 45 classes
batch_size = 64 
classes_num = 45
epochs = 25
lr = .001

'''
Bottom Model: SqueezeNet without top
'''

model = sqznet_models.SqueezeNet(include_top = False)
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )

test_data = model.predict_generator(test_generator, 3150 // batch_size + 1)

'''
Top Model
'''
top_model = load_model(model_path)
top_model.load_weights(model_weights_path)

test_labels = np.arange(classes_num)
test_labels = np.repeat(test_labels, 70)

test_preds = top_model.predict(test_data)
test_preds = np.argmax(test_preds, axis=-1)

class_list = ['']
for subdir, dirs, files in os.walk('./data'):
  if subdir.startswith('./data/train/'):
    name = subdir[13:]
    if name.startswith('.'):
        continue
    class_list.append(name)

class_list = class_list[1:]
conf = metrics.confusion_matrix(test_labels, test_preds)

recall = [0] * classes_num
precision_predicted = [0] * classes_num
precision_actual = [0] * classes_num

for i in range(0, len(test_preds)):
    if test_preds[i] == test_labels[i]:
        recall[i//70] += 1
        precision_actual[test_preds[i]] += 1
    precision_predicted[test_preds[i]] += 1

precision_stats = []
for i in range(0, classes_num):
    if precision_predicted[i] > 0:
        precision_stats.append(precision_actual[i] / precision_predicted[i])
    else:
        precision_stats.append(0)
print(precision_stats)

print("Average recall: {}".format(str(float(sum(recall)) / len(recall)  / classes_num )))
print("Average precision: {}".format(str(float(sum(precision_stats)) / len(precision_stats) / classes_num )))


'''
Code to print confusion matrix, sourced from the cited sklearn website

(https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

Was run with python2.

'''
plt.figure()

def plot_confusion_matrix(conf, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='binary'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Code for printing numbers in squares
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


plot_confusion_matrix(conf, class_list, normalize=True, title='Normalized Confusion Matrix')
plt.show()


