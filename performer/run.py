import os
from queue import Queue

import cv2
import keras
import numpy as np
import rasterio
import tensorflow as tf
import tqdm
from rasterio import Affine
from rasterio.windows import Window








###############################################################################
#   Here are three example model submissions.
#
#   Each model takes an OpenCV image and the name of the
#   input image (not the path). The model then performs
#   some operation on the image and saves to the designated
#   output path. The output paths are where the organizers
#   will receive the results of each model. An empty output
#   folder will signify there was no submission for that
#   that model. You must submit one model and have it write
#   its output to /root/output0. DO NOT MODIFY.
#
#   How each model handles its IO, either from input
#   source / to output or its method calls is up to the
#   discretion of the performer. See description in
#   __main__ for some IO optimization for Docker caching
#   recommendations and an example of it.
#
#   Absolute Requirements:
#   1.  A minimum of one model must be submitted
#   2.  The model must load all input images from /root/input
#   3.  The model must modify and save a new copy of all input images
#   4.  All image names must remain the same.
#       The image compression/extension can be changed to any
#       Ubuntu supported .jpeg, .jpg, .tiff, .tif, .png format.
###############################################################################


model = keras.models.load_model('superres_v3.0_2018-01-25 22:38:29.338372.h5', custom_objects={"tf":tf})

#   An example model that makes no changes to the input image
def model0(img, name):
    save = os.path.join('.', 'root', 'output0', name)
    print(save)
    red_chan = model.predict(np.expand_dims(img, 0)[:,:,0:1])
    blu_chan = model.predict(np.expand_dims(img, 0)[:,:,1:2])
    gre_chan = model.predict(np.expand_dims(img, 0)[:,:,2:])
    new_img = np.stack([red_chan, blu_chan, gre_chan], axis=-1)
    print(new_img.shape)

    cv2.imwrite(save, new_img)

#   An example model that performs a median blur on the input image
def model1(img, name):
    save = os.path.join('', 'root', 'output1', name)

    cv2.medianBlur(img, 3)

    cv2.imwrite(save, img)

#   An example model that switches channel order
#   and changes image compression on the input image
def model2(img, name):
    save = os.path.join('', 'root', 'output2', name)

    red = img[:,:,2].copy()
    blue = img[:,:,0].copy()
    img[:,:,0] = red
    img[:,:,2] = blue

    cv2.imwrite(save.split('.')[0]+'.tif', img)




if __name__ == '__main__':
    #   This is the path to where the input images will be when
    #   this container is executed by the organizers. Changing
    #   this path will fail to produce results. DO NOT CHANGE.
    #input_ = '/root/input'
    input_ = './root/input'

    paths = os.listdir(input_)

    #   In this example, the loop iterates over all of the input
    #   images and loads the image and then provides a copy
    #   to each model. Other image IO methods and model function
    #   calls can be used. E.g. pass image path to the model.
    #
    #   If a model can perform on one image at a time without
    #   effecting the performance, then this style is the optimal
    #   loading paradim for Docker caching rather than doing the
    #   entire input image set for a model, then repeating for
    #   subsequent models.
    for p in paths:
        img = cv2.imread(os.path.join(input_, p))

        #   Submit a minimum of one model, but a
        #   second and third model is optional.
        model0(np.copy(img), p)
        #model1(np.copy(img), p)   # uncomment to use more than one model
        #model2(np.copy(img), p)   # uncomment to use more than one model

