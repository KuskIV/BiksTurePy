import numpy as np
import random as rd
import re
import  tensorflow as tf
from PIL import Image
import random

def changeImageSize(maxWidth: int, 
                    maxHeight: int, 
                    image:np.array):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    return image.resize((newWidth, newHeight))

def EnsureUniformImageShape(img1: Image.Image,img2: Image.Image, shape=None):
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
                     shape:tuple = None):
    image3,image4 = EnsureUniformImageShape(img1, img2,shape=shape)

    image5 = image3.convert("RGBA")
    image6 = image4.convert("RGBA")

    return Image.blend(image5, image6, alpha=alpha)

def convertToPILImg(img1: np.array, normilized = True)-> Image.Image:

    if normilized is True:
        img1 = img1 * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(img1, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    return im

def get_class_names()->list:
    """Gets classification name for each label from the labels.txt, which is assumed to be in root."""
    class_names = []
    # RegEx to match for labels in labels.txt
    labels_regex = re.compile('(?<== )(.)*')

    with open('labels.txt', 'r') as fp:
        i = 0
        for line in fp:
            match = labels_regex.search(line).group(0)
            class_names.append(match)

    return class_names

def make_prediction(model, image): # hopfully delete
    img_reshaped = tf.reshape(image, (1, 32, 32, 3))
    return model.predict_step(img_reshaped)

def display_ppm_image(path: str)->None:
    """"Input a path to original image to display it"""
    im = Image.open(path)
    im.show()

def display_numpy_image(numpy_image:np.array)->None:
    """Input a (0 to 1) normalized numpy representation of a PIL image, to show it"""
    # Scaling the pixels back
    numpy_image_rescaled = numpy_image * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(numpy_image_rescaled, np.uint8)
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
        im_array = np.asarray(im_ppm) # Convert to numpy array
        converted_images.append(im_array / 255.0) # Normalize pixel values to be between 0 and 1

    return converted_images

def Shuffle(img_dataset, img_labels):
    img_dataset_in = img_dataset
    img_labels_in = img_labels

    z = zip(img_dataset, img_labels)
    z_list = list(z)
    random.shuffle(z_list)
    img_dataset_tuple, img_labels_tuple = zip(*z_list)
    img_dataset_in = np.array(img_dataset_tuple)
    img_labels_in = np.array(img_labels_tuple)

    return img_dataset_in, img_labels_in