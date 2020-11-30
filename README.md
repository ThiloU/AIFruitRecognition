# AIFruitRecognition
Recognising different kinds of fruits and vegetables on photos using Python3 and Google's machine learning platform "Tensorflow"

## Running instructions
 There are several python packages required for this project.
 You can install them via the commandline using pip3:

- ```pip3 install matplotlib```
- ```pip3 install Pillow```
- ```pip3 install tensorflow==2.3.1``` (to force version 2.3.1)
- ```pip3 install scipy```
- ```pip3 install playsound```
- ```pip3 install opencv-python```
- ```pip3 install flask```


All other dependencies should be pre-installed

Now you have to clone the repository into a folder. You can optinally unpack [this](https://drive.google.com/file/d/1ZX6YbiFL6fds2Evkfod1Wt4jWUdMnjyf/view?usp=sharing) file containing the training data.

### Running as App on a phone
This way, the software is the most easy to use for the user, but it will require you to do some changes, for example the ip-address in the app.

First, you will have to modify the app part of the project to match the ip-address of your PC/server. For that, go to the [MIT App Inventor]( http://ai2.appinventor.mit.edu/)(You have to log in with a google account). Then, cick on "Projects", and then on "Import project(.aia) from my computer...". In the dialog, choose the "ProfilApp.aia" file that came with this repository.
To change the ip-addresses of the webserver and the redis-server, click on the "Web1"/"CloudDBUse" components under the shown phone screen. On the right side of the App Inventor, you can now see the properties of the component and you can change the ip-address. Then, to get the app as .apk file, click on "Build" on the top of the screen and choose a option that works for you.

The next step is to start the python script ```predictionServer.py```.
This will start a webserver that will listen for incoming requests from the app and will send the prediction back to the device.
This script has to have an "uploads" folder to store the images in, and a "data" folder, which you can copy from this repository.

You also need to start a Redis CloudDB server to accept the photos uploaded from the app.
To do that, [download Redis](https://redis.io/download) and run it using this command: ```redis-server --protected-mode no --requirepass "foo"```.
The additional parameters allow the user to upload data to the database using the password "foo".

You can the connect your phone to the network of your PC and start the app. Everything should now work.

### Running as python script on the PC
To use the program on the PC, you can start ```predictFromCam.py```. In the upcoming window, you can press "s" to save the picture and analyse it.
