# AIFruitRecognition
Recognising different kinds of fruits and vegetables on photos using Python3 and Google's machine learning platform "Tensorflow"

## Running Instructions
 There are several python packages required for this project.
 You can install them via the commandline using pip3:
<<<<<<< HEAD
-        pip3 install matplotlib
-        pip3 install Pillow
-        pip3 install tensorflow
-        pip3 install scipy
-        pip3 install opencv-python
=======
- ```pip3 install matplotlib```
- ```pip3 install Pillow```
- ```pip3 install tensorflow==2.3.1``` (to force version 2.3.1)
- ```pip3 install scipy```
>>>>>>> 866fefbe11d15e29b8987c2458c41945ea150543

All other dependencies should be pre-installed

Now you have to clone the repository into a folder and also unpack [this](https://drive.google.com/file/d/1NjBw3OjpYbcKbrk9fxnqXcKJwJHDdFql/view?usp=sharing) file containing the training data.

 There is already a pre-learned model included, so you can just execute the file "fruitClassificationAugmentationDropoutModelLoadAndPredict.py" using this command:
<<<<<<< HEAD
```python3 fruitClassificationModelLoadAndPredictFromFile.py```
To input your own pictures for prediction, simply change line 20: ```predictionPath = "./predictionPhotos/appleOwn.jpg"``` and give a path to your own picture.


If you want to create your own model, take a look at the file "fruitClassificationModelSave.py"
=======
```python3 fruitClassificationAugmentationDropoutModelLoadAndPredict.py```

To input your own pictures for prediction, simply change line 20: ```predictionPath = "./predictionPhotos/appleOwn.jpg"``` and give a path to your own picture.


If you want to create your own model, take a look at the file "fruitClassificationAugmentationDropoutModelSave.py": 
In line 14 you can change the directory with the training pictures and
in line 15 you can change the number of epochs you want to train your model for (the more epochs, the better the finished model).

You can then start the program using ```python3 fruitClassificationAugmentationDropoutModelSave.py```
After the program has finished, it will write the model in the ./savedModel directory where you can use it for future Predictions
>>>>>>> 866fefbe11d15e29b8987c2458c41945ea150543
