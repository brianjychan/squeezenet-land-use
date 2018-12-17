"""
Baseline model is Keras 2.0, adapted from Keras 1.0 "CNN Image Classifier" from Github user tatsuyah at https://github.com/tatsuyah/CNN-Image-Classifier.

Code modified to accomodate scalable dataset building and directory walking.
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from sklearn.metrics import f1_score

img_width, img_height = 64, 64
model = "3lay"
model_path = './models/{}.h5'.format(model)
model_weights_path = './models/{}weights.h5'.format(model)

# model_path = './models/caffe-2018-12-14 04:40:14.558896.h5'
# model_weights_path = './models/caffeweights2018-12-14 04:40:14.725004.h5'

model = load_model(model_path)
model.load_weights(model_weights_path)

# class_list = ['']
# for subdir, dirs, files in os.walk('./data'):
#   if subdir.startswith('./data/train/'):
#     name = subdir[13:]
#     class_list.append(name)


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  x = x / 255

  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  # print(x.shape)
  # print(array)
  return answer

y_actual = []
y_pred = []

recall = [0] * 45 # correctly predicted an i was i

precision_predicted = [0] * 45 # predicted to have i
precision_actual = [0] * 45 # actually had i when predicted to


for root, dirs, files in os.walk('./data/test/'):
  for i, dr in enumerate(sorted(dirs)):
    # print(dr)
    # print(i)
    beg = root + dr + '/'
    for root2, dirs2, files2 in os.walk(root + dr + '/'): #walk this dir
      for filename in files2: #get filename
        if filename.startswith("."):
          continue
        
        result = predict(beg + filename)

        y_actual.append(i)
        y_pred.append(result)

        if result == i:
          recall[i] += 1
          precision_predicted[i] += 1
          precision_actual[i] += 1
        else:
          precision_predicted[result] += 1

    # if recall[i] == 0:
    #   print("0 recall: {}".format(dr))
          

print(recall)
tot = []
for i in range(0, 45):
  tot.append(precision_actual[i] / precision_predicted[i])
print(tot)

print("F1")
print(f1_score(y_actual, y_pred, average='macro'))

