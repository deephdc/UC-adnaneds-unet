

# OS and others packages
import os


import numpy as np
import matplotlib.pyplot as plt


# image processing libraries

import skimage.io as io
import cv2

from skimage import transform
from skimage import img_as_bool


# Tensorflow packages 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from focal_loss import BinaryFocalLoss



# Some parametres for preprocessing image_input

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

input_size_2 = (IMG_WIDTH, IMG_HEIGHT)
input_size_3 = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)






######## Preprocessing 

input_image 

read_image = io.imread(path)
image_resized = transform.resize(read_image, input_size_2).astype(np.float32)


def do_image_segmentation(
    layer: ImageData
    ) -> ImageData:
    
    def redimension(image):
        X = np.zeros((1,1024,1024,3),dtype=np.uint8)
        size_ = image.shape
        img = image[:,:,:3]
        X[0] = resize(img, (1024, 1024), mode='constant', preserve_range=True)
        return X,size_


    
    image_reshaped,size_ = redimension(layer)

    #path_0 = os.path.abspath("napari-segmentation")
    path_0 = os.getcwd() 
    full_path = os.path.join(path_0, "napari-segmentation/src/napari_segmentation", "best_model_docker_nap/best_model_docker_nap.h5")

    model_new = tf.keras.models.load_model(full_path, custom_objects={'dice_coefficient': dice_coefficient})
    prediction = model_new.predict(image_reshaped)
    preds_test_t = (prediction > 0.30000000000000004).astype(np.uint8)
    temp = np.squeeze(preds_test_t[0,:,:,2])*255

    return cv2.resize(temp, dsize=(size_[1],size_[0]))





############# Prediction 


## Metrics for prediction 

def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 




## Load Model for prediction  