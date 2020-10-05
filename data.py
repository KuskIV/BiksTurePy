import tensorflow as tf
import numpy
from PIL import Image
import extract

import random
import math


# DATASET_PATH = 'C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images' # assume it is in root
DATASET_PATH = 'images/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
#DATASET_PATH = 'C:\\Users\\jeppe\\Desktop\\FullIJCNN2013'
#DATASET_PATH = 'FullIJCNN2013'

def display_ppm_image(path: str)->None:
    """"Input a path to original image to display it"""
    im = Image.open(path)
    im.show()

def display_numpy_image(numpy_image:numpy.array)->None:
    """Input a (0 to 1) normalized numpy representation of a PIL image, to show it"""
    # Scaling the pixels back
    numpy_image_rescaled = numpy_image * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = numpy.array(numpy_image_rescaled, numpy.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    im.show()

def convert_imgs_to_numpy_arrays(dataset: list)->list:
    """Receive a dataset in and return an numpy array of the images
       converted to normalized (0 to 1) numpy arrays."""
    converted_images = []
    # Convert images to numpy arrays
    for image in dataset:
        im_ppm = Image.open(image[0]) # Open as PIL image
        im_array = numpy.asarray(im_ppm) # Convert to numpy array
        converted_images.append(im_array / 255.0) # Normalize pixel values to be between 0 and 1

    return converted_images


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


def get_data(fixed_size:tuple=(0,0), padded_images:bool = False, smart_resize:bool = True)->tuple:
    # extract data from raw
    raw_dataset, images_per_class = extract.Updated_GetData(DATASET_PATH) #using old get_data function temporarily as, new does not work for FullIJCNN2013

    if padded_images:
        print("Padded images not implemented yet, only resize and smart resize.")

    # convert ppm to numpy arrays
    numpy_images = convert_imgs_to_numpy_arrays(raw_dataset)
    # auto reshape images
    numpy_images_reshaped = auto_reshape_images(fixed_size, numpy_images, smart_resize)

    # get labels for each training example in correct order
    labels = get_labels(raw_dataset)

    return numpy_images_reshaped, labels, images_per_class

def Shuffle(img_dataset, img_labels):
    img_dataset_in = img_dataset
    img_labels_in = img_labels

    z = zip(img_dataset, img_labels)
    z_list = list(z)
    random.shuffle(z_list)
    img_dataset_tuple, img_labels_tuple = zip(*z_list)
    img_dataset_in = numpy.array(img_dataset_tuple)
    img_labels_in = numpy.array(img_labels_tuple)

    return img_dataset_in, img_labels_in

def split_data(img_dataset:list, img_labels:list, images_per_class, training_split:float=.7, shuffle:bool=True)->tuple:
    """Input numpy array of images, numpy array of labels.
       Return a tuple with (training_images, training_labels, test_images, test_labels).
       Does have stochastic/shuffling of the data with shuffle parameter."""


    train_set = []
    train_label = []

    val_set = []
    val_label = []

    #for i in range(len(images_per_class)):
    #    print(images_per_class[i])


    label_index = 0
    maxVal = images_per_class[label_index]
    pictures_in_current_class = images_per_class[label_index]
    dist_in_current_class = pictures_in_current_class * training_split
    #print(f"Class: {label_index}, pictures in class: {pictures_in_current_class}, dist in class: {dist_in_current_class}")
    val_in_train = 0
    val_in_eval = 0
    for i in range(len(img_dataset)):
        if i > maxVal:
            #print(f"Train size: {val_in_train}, Evaluation size: {val_in_eval}, overall: {val_in_train + val_in_eval}, dist = {dist_in_current_class}")
            label_index += 1
            maxVal = images_per_class[label_index] + i
            val_in_train = 0
            val_in_eval = 0
            if not (label_index > len(images_per_class)):
                pictures_in_current_class = images_per_class[label_index]
                dist_in_current_class = math.ceil(pictures_in_current_class * training_split)
                #print(f"Class: {label_index}, pictures in class: {pictures_in_current_class}, dist in class: {dist_in_current_class}")
        if val_in_train < dist_in_current_class:
            train_label.append(img_labels[i])
            train_set.append(img_dataset[i])
            val_in_train += 1
        else:
            val_label.append(img_labels[i])
            val_set.append(img_dataset[i])
            val_in_eval += 1

    if shuffle:
        val_set, val_label = Shuffle(val_set, val_label)
        train_set, train_label = Shuffle(train_set, train_label)

    return train_set, train_label, val_set, val_label

    """
    img_dataset_in = img_dataset
    img_labels_in = img_labels

    if shuffle:
        z = zip(img_dataset, img_labels)
        z_list = list(z)
        random.shuffle(z_list)
        img_dataset_tuple, img_labels_tuple = zip(*z_list)
        img_dataset_in = numpy.array(img_dataset_tuple)
        img_labels_in = numpy.array(img_labels_tuple)

    num_of_examples = img_dataset.shape[0]
    split_pivot = math.floor(num_of_examples * training_split) # floor

    training_images = numpy.array([img_dataset_in[i] for i in range(split_pivot)])
    training_labels = numpy.array([img_labels_in[i] for i in range(split_pivot)])
    test_images = numpy.array([img_dataset_in[i] for i in range(split_pivot, num_of_examples)])
    test_labels = numpy.array([img_labels_in[i] for i in range(split_pivot, num_of_examples)])

    return training_images, training_labels, test_images, test_labels
    """
