# CNN_dogcat_classifier

**This is a compilation of attempts to create CNNs to classify whether images contain either dogs or cats.**
*The 'models' directory containing all the trained neural network models and weights is excluded in the repository because of size, but can be found here:* (https://drive.google.com/drive/folders/1quS8edFfFNworRGatKCiGTeFmVXTipxg?usp=sharing)

Small comments about version updates are included at the top of each Jupyter notebook.

The most succesful model was resnetv3, which achieved a final accuracy of 0.99 with the validation data.

*Data used for training was taken from the Kaggle collection of dog and cat pictures found here:* (https://www.kaggle.com/c/dogs-vs-cats/data)

**Also included is the python program resnet_cam.py which uses your camera to detect whether a dog/cat is in frame using the resnetv3 model**
Be default, the program uses the camera at 50fps, with the model making predictions once every frame.
