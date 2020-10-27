import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 => log everything; 1 => no INFO; 2 => no INFO/WARNINGS; 3 => no INFO/WARNINGS/ERRORS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import PIL
import pathlib
from operator import itemgetter
import numpy as np
import json

import cv2
from playsound import playsound


key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)


def getImage():
    while True:
        check, frame = webcam.read()
        # print(check) #prints true as long as the webcam is running
        # print(frame) #prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            print("Saving image...")
            cv2.imwrite(filename=imgSavePath, img=frame)
            print("Saved image")
            webcam.release()
            img_new = cv2.imread(imgSavePath)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            break


img_height = 100
img_width = 100

imgSavePath = "./data/saved_img.jpg"
classNamesPath = "./data/classes.json"
predictionPath = "./data/saved_img.jpg"
modelPath = "./data/savedModel"

with open(classNamesPath, "r") as f:
    class_names = json.loads(json.load(f))

getImage()

model = tf.keras.models.load_model(modelPath)

img = keras.preprocessing.image.load_img(
    predictionPath, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

fruitName = class_names[np.argmax(score)]
probability = 100 * np.max(score)
fruit = "./audio/" + fruitName + '.mp3'
notSure = "./audio/notSure.mp3"
connectingOr = "./audio/or.mp3"

classAndProb = []
score = score.numpy()
for num, name in enumerate(class_names):
    classAndProb.append([name, "{:.2f}".format(float(score[num])*100)])
classAndProb = sorted(classAndProb, key=itemgetter(1), reverse=True)

fruitSecondChoice = "./audio/" + classAndProb[1][0] + '.mp3'


print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(fruitName, probability))

if(probability >= 90):
    playsound(fruit)
else:
    playsound(notSure)
    playsound(fruit)
    playsound(connectingOr)
    playsound(fruitSecondChoice)
