"""
Baseline model is Keras 2.0, adapted from Keras 1.0 "CNN Image Classifier" from Github user tatsuyah at https://github.com/tatsuyah/CNN-Image-Classifier.

Code modified to accomodate scalable dataset building and directory walking.
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 64, 64
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

class_list = ['']
for subdir, dirs, files in os.walk('./data'):
  if subdir.startswith('./data/train/'):
    name = subdir[13:]
    class_list.append(name)


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  return answer

count_correct = [0] * 45
count_wrong = [0] * 45



for i, ret in enumerate(os.walk('./data/validation/')):
  if i ==0:
    continue
  print("next category")
  for j, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue 
    result = predict(ret[0] + '/' + filename)
    if j < 8:
      print(ret[0], filename)
      print("result: {}".format(result))
