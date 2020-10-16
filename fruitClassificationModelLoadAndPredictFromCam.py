import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 => log everything; 1 => no INFO; 2 => no INFO/WARNINGS; 3 => no INFO/WARNINGS/ERRORS

import PIL
import tensorflow as tf
import pathlib
from operator import itemgetter

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def getImage():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    print("Found webcam")
    while True:
        check, frame = webcam.read()
        #print("Is the webcam running? " + str(check)) #prints true as long as the webcam is running
        #print(frame) #prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg')
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            break


img_height = 100
img_width = 100

data_dir = "./trainingSmaller"
getImage()
predictionPath = "saved_img.jpg"

classes = tf.keras.preprocessing.image_dataset_from_directory(data_dir)
class_names = classes.class_names

model = tf.keras.models.load_model('./savedModel')

img = keras.preprocessing.image.load_img(
    predictionPath, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

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
    print("Content: {} -- Probability {}%".format(i[0],i[1]) )
