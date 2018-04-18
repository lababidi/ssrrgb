import os

import cv2
import numpy as np

import keras

import tensorflow as tf

import scipy as sp


def SubpixelConv2D(scale=4, name="subpixel"):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param name:
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(in_shape):
        ret_shape1 = None if in_shape[1] is None else in_shape[1]*scale
        ret_shape2 = None if in_shape[2] is None else in_shape[2]*scale
        return tuple([in_shape[0], ret_shape1, ret_shape2, int(in_shape[3] // (scale*scale))])

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)





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


model = keras.models.load_model('m1.h5', custom_objects={"tf":tf})

#   An example model that makes no changes to the input image
def model0(img, name):
    save = os.path.join('.', 'root', 'output0', name)
    new_img = model.predict(np.expand_dims(img, 0))[0][0]
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

