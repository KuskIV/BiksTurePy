import numpy as np
import random as rd
import re
import  tensorflow as tf
from PIL import Image
import random

def changeImageSize(maxWidth: int, 
                    maxHeight: int, 
                    image:Image.Image)->Image.Image:
    """Resizes an image

    Args:
        maxWidth (int): The max width of the image
        maxHeight (int): The max heigth of the image
        image (np.array): The input image to resize

    Returns:
        Image.Image: The resized image
    """
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    return image.resize((newWidth, newHeight))

def EnsureUniformImageShape(img1: Image.Image,img2: Image.Image, shape=None)->tuple:
    """This method ensure two imagse are uniform, and returns them

    Args:
        img1 (Image.Image): The first image
        img2 (Image.Image): The second image
        shape ([type], optional): The shape to set both image to. Defaults to None.

    Returns:
        tuple: A tuple containing both images
    """

    size1 = img1.size
    size2 = img2.size

    if(shape != None):
        img1 = changeImageSize(shape[0], shape[1], img1)
        img2 = changeImageSize(shape[0], shape[1], img2)
        return img1,img2
    elif(size2 == size1):
        return img1,img2
    else:
        img2 = changeImageSize(size1[0],size1[1],img2)
        return img1,img2

def merge_two_images(img1: Image.Image
                    ,img2: Image.Image,
                     alpha:float=0.25, 
                     shape:tuple = None)->Image.Image:
    """Merges two images, and returns one image

    Args:
        img1 (Image.Image): The first image
        img2 (Image.Image): The second imaeg
        alpha (float, optional): The opacity of the imaeg when merging them. Defaults to 0.25.
        shape (tuple, optional): The shape of the output image. Defaults to None.

    Returns:
        Image.Image: The two images merged
    """
    image3,image4 = EnsureUniformImageShape(img1, img2,shape=shape)

    image5 = image3.convert("RGBA")
    image6 = image4.convert("RGBA")

    return Image.blend(image5, image6, alpha=alpha)

def convertToPILImg(img1: np.array, normilized = True)-> Image.Image:
    """Converts a numpy array to a PIL image

    Args:
        img1 (np.array): The array to convert
        normilized (bool, optional): If the value should be between 0-1. Defaults to True.

    Returns:
        Image.Image: The output image after it has been converted
    """

    if normilized is True:
        img1 = img1 * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(img1, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    return im

def get_class_names()->list:
    """Gets classification name for each label from the labels.txt, which is assumed to be in root.
    
    Returns:
        list: A list of all the class names
    """
    class_names = []
    # RegEx to match for labels in labels.txt
    labels_regex = re.compile('(?<== )(.)*')

    with open('labels.txt', 'r') as fp:
        i = 0
        for line in fp:
            match = labels_regex.search(line).group(0)
            class_names.append(match)

    return class_names

# def make_prediction(model, image): # hopfully delete
#     img_reshaped = tf.reshape(image, (1, 32, 32, 3))
#     return model.predict_step(img_reshaped)

def display_ppm_image(path: str)->None:
    """Input a path to original image to display it

    Args:
        path (str): The path to the original image
    """
    im = Image.open(path)
    im.show()

def convert_numpy_image_to_image(numpy_image:np.array):
    # Scaling the pixels back
    numpy_image_rescaled = numpy_image * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(numpy_image_rescaled, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    return im

def display_numpy_image(numpy_image:np.array)->None:
    """Input a (0 to 1) normalized numpy representation of a PIL image, to show it

    Args:
        numpy_image (np.array): The input image to normalize
    """

    im = convert_imgs_to_numpy_arrays(numpy_image)
    im.show()




def convert_imgs_to_numpy_arrays(dataset: list)->list:
    """Receive a dataset in and return an numpy array of the images
       converted to normalized (0 to 1) numpy arrays.

    Args:
        dataset (list): A list of PIL images to convert

    Returns:
        list: A list of numpu arrays after they have been converted
    """
    converted_images = []
    # Convert images to numpy arrays
    for image in dataset:
        im_ppm = Image.open(image[0]) # Open as PIL image
        im_array = np.asarray(im_ppm) # Convert to numpy array
        converted_images.append(im_array / 255.0) # Normalize pixel values to be between 0 and 1

    return converted_images

def auto_reshape_images(fixed_size: tuple, numpy_images: list, smart_resize:bool = True)->np.array:
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

    return np.array(reshaped_images)
