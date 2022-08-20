# A program that uses the computer webcam to determine whether a dog/cat is in frame
# Uses the resnetv3 trained model

import numpy as np
import os
import cv2
import tensorflow
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.applications.resnet50 import ResNet50

IMG_SIZE = 300
IMG_DIM = (IMG_SIZE, IMG_SIZE)
FPS = 50 #set to 0 for individual frames controlled by key presses
# 50fps = 1 frame per 20ms

# model setup

model = keras.models.load_model('models/dogsvscats-2e-05-resnet.model')
# commented out for now because it takes time to load

# functions for model predictions

# NOTE: predictions usually take less than 20ms to process

def get_photo(photo_arr):
    train_imgs = []
    res = cv2.resize(photo_arr, IMG_DIM)
    train_imgs.append(res)
    process = np.array(train_imgs)
    process = tensorflow.keras.applications.resnet50.preprocess_input(process)
    return process

def predict(path):
    return model.predict( get_photo(path), verbose=0 )

cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    prediction = np.float32(predict(frame)[0][0])
    text = "Error"

    if(prediction > 0.5):
        text = "I see a dog (" + str(round(prediction,1)) + ")"
    else:
        text = "No dog here (" + str(round(prediction,1)) + ")"

    text_coordinates = (50,75)
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 2
    fontColor = (255,255,255)
    font_thickness = 2
    frame = cv2.putText(frame, text, text_coordinates, font, fontScale, fontColor, font_thickness)

    cv2.imshow("Dog/Cat Classifier - Press escape to exit", frame)
    

    key = cv2.waitKey(round(1000/FPS))
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
