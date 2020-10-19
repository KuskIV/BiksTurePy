import tensorflow as tf
import numpy
from PIL import Image
import os
import math
from global_paths import get_dataset_path




def get_labels(dataset: list)->numpy.array:
    """Input is the dataset in list of [[image_path, label]].
       Returns a numpyarray of uint8 of the labels."""
    labels = []
    for image in dataset:
        labels.append(image[1])

    return numpy.array(labels, dtype=numpy.uint8)