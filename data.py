import tensorflow as tf
import numpy
from PIL import Image
import os
import math
from global_paths import get_dataset_path

def auto_reshape_images(fixed_size: tuple, numpy_images: list, smart_resize:bool = True)->numpy.array:
    """Reshapes the entire dataset in the minimal needed reshaping, by reshaping
       to max width in the dataset and max height. Default Uses tf.keras.preprocessing.image.smart_resize
       to do the actual reshaping in order (input to that is numpy array representaion of images), otherwise
       can use standard resize """
    max_width = 0
    max_height = 0
    reshaped_images = []

    # find max width and height
    for image in numpy_images:
        if len(image) > max_width:
            max_width = len(image)
        if len(image[0]) > max_height:
            max_height = len(image[0])

    reshape_size = (max_width, max_height)
    print("reshape size")
    print(reshape_size)


    if sum(fixed_size): # check if fixed size option
        reshape_size = fixed_size

    for image in numpy_images:
        if smart_resize:
            reshaped_images.append(tf.keras.preprocessing.image.smart_resize(image, reshape_size))
        else:
            reshaped_images.append(tf.image.resize(image, reshape_size))

    return numpy.array(reshaped_images)


def get_labels(dataset: list)->numpy.array:
    """Input is the dataset in list of [[image_path, label]].
       Returns a numpyarray of uint8 of the labels."""
    labels = []
    for image in dataset:
        labels.append(image[1])

    return numpy.array(labels, dtype=numpy.uint8)