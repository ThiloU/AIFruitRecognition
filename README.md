# AIFruitRecognition
Recognising different kinds of fruits and vegetables on photos using Python3 and Google's machine learning platform "Tensorflow"

## Running Instructions
 There are several python packages required for this project.
 You can install them via the commandline using pip3:
-        pip3 install matplotlib
-        pip3 install Pillow
-        pip3 install tensorflow
-        pip3 install scipy
-        pip3 install opencv-python

All other dependencies should be pre-installed

Now you have to clone the repository into a folder and unpack the .zip file containing the training data.

 There is already a pre-learned model included, so you can just execute the file "fruitClassificationAugmentationDropoutModelLoadAndPredict.py" using this command:
```python3 fruitClassificationModelLoadAndPredictFromFile.py```
To input your own pictures for prediction, simply change line 20: ```predictionPath = "./predictionPhotos/appleOwn.jpg"``` and give a path to your own picture.


If you want to create your own model, take a look at the file "fruitClassificationModelSave.py"
