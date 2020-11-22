import os
from flask import Flask, request, send_from_directory
from werkzeug import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 => log everything; 1 => no INFO; 2 => no INFO/WARNINGS; 3 => no INFO/WARNINGS/ERRORS
import tensorflow as tf
from tensorflow import keras

from operator import itemgetter
import json

import base64
import redis



UPLOAD_FOLDER = 'uploads'

modelPath = "./data/savedModel"
classNamesPath = "./data/classes.json"
model = tf.keras.models.load_model(modelPath)

r = redis.Redis(host="192.168.1.12", port=6379, password="foo")


def analyseFile(filename):
    img_height = 100
    img_width = 100

    predictionPath = uploadFolder + '/' + filename
    modelPath = "./data/savedModel"
    classNamesPath = "./data/classes.json"

    with open(classNamesPath, "r") as f:
        class_names = json.loads(json.load(f))

    model = tf.keras.models.load_model(modelPath)

    img = keras.preprocessing.image.load_img(
        predictionPath, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    classAndProb = []
    score = score.numpy()
    for num, name in enumerate(class_names):
        classAndProb.append([name, "{:.2f}".format(float(score[num])*100)])
    classAndProb = sorted(classAndProb, key=itemgetter(1), reverse=True)
    print("Returning: " + str(classAndProb))
    return classAndProb


def decodeToFile(keyName):
    dbString = r.get("profilApp:" + keyName)
    jsonString = dbString.decode("utf-8")
    jsonObject = json.loads(jsonString)
    codedString = jsonObject[1]

    imgdata = base64.decodestring(codedString.encode("utf-8"))
    # imgdata = base64.b64decode(codedString)
    filePath = './uploads/{}.jpg'.format(keyName)

    with open(filePath, 'wb') as f:
        f.write(imgdata)

    return keyName + ".jpg"


app = Flask(__name__)
uploadFolder = "./uploads"
app.config['UPLOAD_FOLDER'] = uploadFolder


@app.route('/', methods=['GET', 'POST'])
def get_file():
    if request.method == 'POST':
        keyName = request.form['keyName']
        print(keyName)
        fileName = decodeToFile(keyName)

        response = analyseFile(fileName)
        return json.dumps(response)


@app.route('/myip', methods=['GET'])
def get_ip():
    if request.method == 'GET':
        userIp = request.remote_addr
        print("New user connected with IP: " + userIp)
        return userIp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080')
