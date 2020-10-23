import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 => log everything; 1 => no INFO; 2 => no INFO/WARNINGS; 3 => no INFO/WARNINGS/ERRORS

import PIL
import tensorflow as tf
import pathlib
from operator import itemgetter

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

img_height = 100
img_width = 100

data_dir = "./pictures/trainingSmaller"
#predictionPath = "./pictures/predictionPhotosFruits/bananaOwn2.jpg"
predictionPath = "./data/saved_img.jpg"
modelPath = "./data/savedModel"

classes = tf.keras.preprocessing.image_dataset_from_directory(data_dir)
class_names = classes.class_names

model = tf.keras.models.load_model(modelPath)

# raw_model = tf.keras.models.load_model('./savedModel')
# model = tf.keras.Sequential([raw_model, tf.keras.layers.Softmax()])

img = keras.preprocessing.image.load_img(
    predictionPath, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
classAndProb = []
score = score.numpy()
for num, name in enumerate(class_names):
    classAndProb.append([name, "{:.2f}".format(float(score[num])*100)])
classAndProb = sorted(classAndProb, key=itemgetter(1), reverse=True)
print("")
for i in classAndProb:
    print("Content: {} -- Probability {}%".format(i[0], i[1]))
