import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 => log everything; 1 => no INFO; 2 => no INFO/WARNINGS; 3 => no INFO/WARNINGS/ERRORS

import PIL
import tensorflow as tf
import pathlib
from operator import itemgetter
import json


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

img_height = 100
img_width = 100

predictionPath = "./pictures/predictionPhotosFruits"
modelPath = "./data/savedModel"
classNamesPath = "./data/classes.json"

with open(classNamesPath, "r") as f:
    class_names = json.loads(json.load(f))

model = tf.keras.models.load_model(modelPath)

# raw_model = tf.keras.models.load_model('./savedModel')
# model = tf.keras.Sequential([raw_model, tf.keras.layers.Softmax()])

imgList = os.listdir(predictionPath)
# print(imgList)
for i in imgList:
    img = keras.preprocessing.image.load_img(
        predictionPath + '/' + i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "'{}' most likely belongs to {} with a {:.2f} percent confidence."
        .format(i, class_names[np.argmax(score)], 100 * np.max(score))
    )
