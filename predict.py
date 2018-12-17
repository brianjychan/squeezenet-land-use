"""
Adapted from "CNN Image Classifier" from Github user tatsuyah at https://github.com/tatsuyah/CNN-Image-Classifier.

Pairs with train.py
Basic but inflexible code for making predictions from a model.
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from sklearn.metrics import f1_score

classes_num = 45
img_width, img_height = 64, 64
model = "3lay"
model_path = './models/{}.h5'.format(model)
model_weights_path = './models/{}-weights.h5'.format(model)

model = load_model(model_path)
model.load_weights(model_weights_path)

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

'''
Calculating the metrics of a model for a test directory 

data/
  test/
    class1/
      class1_1.jpg
      ...
    class2/
      class2_1.jpg
    ...
  train/
    ...
  test/
    ...

'''
y_actual = []
y_pred = []

recall = [0] * classes_num # correctly predicted an i was i
precision_predicted = [0] * classes_num # predicted to have i
precision_actual = [0] * classes_num # actually had i when predicted to

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
       

print(recall)

tot = []
for i in range(0, classes_num):
  tot.append(precision_actual[i] / precision_predicted[i])
print(tot)

print("F1")
print(f1_score(y_actual, y_pred, average='macro'))

