import numpy as np
import tensorflow as tf
from PIL import Image
import sys

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import changeImageSize, convertToPILImg, convert_imgs_to_numpy_arrays

def make_prediction(model:tf.python.keras.engine.sequential.Sequential, image:Image.Image, shape=(32, 32, 3))->list:
    """Based on one image, a model makes a prediction to what it is

    Args:
        model (tf.python.keras.engine.sequential.Sequential): The model to use
        image (Image.Image): The imaage to predict on

    Returns:
        list: The probability of the given image beign each class. Softmax not yet applied
    """
    shape = (1, shape[0], shape[1], shape[2])
    #img_reshaped = tf.reshape(image, (1, 32, 32, 3))
    img_reshaped = tf.reshape(image, shape)
    return model.predict_step(img_reshaped)


def partial_accumilate_distribution(test_images:list, test_labels:list, size:tuple, model=None, SAVE_LOAD_PATH=None)->None:
    """Given a set of images and lables, a prediction is made for each image

    Args:
        SAVE_LOAD_PATH (str): The path to load the model from
        test_images (list): A set of images
        test_labels (list): A set of lables
    """
    if model == None:
        if SAVE_LOAD_PATH == None:
                print("partial_accumilate_distribution has been called without a model or a path to a model. The program will now exit.")
                sys.exit()
        model = tf.keras.models.load_model(SAVE_LOAD_PATH)
        model.summary()

    accArr = np.zeros((43, 2))

    for i in range(len(test_images)):
        print(type(test_images[i]), " -----------", test_images[i].size)
        prediction = make_prediction(model, test_images[i], shape=(size[0], size[1], 3))
        softmaxed = tf.keras.activations.softmax(prediction)
        if test_labels[i] == np.argmax(softmaxed):
            accArr[int(test_labels[i])][1] = accArr[int(test_labels[i])][1] + 1
        else:
            accArr[int(test_labels[i])][0] = accArr[int(test_labels[i])][0] + 1
        break

    return accArr

#def print_accumilate_distribution(SAVE_LOAD_PATH:str, test_images:list, test_labels:list, size)->None:
def print_accumilate_distribution(accArr:list, size=None)->None:
    """Given a set of images and lables, a prediction is made for each image

    Args:
        SAVE_LOAD_PATH (str): The path to load the model from
        test_images (list): A set of images
        test_labels (list): A set of lables
    """
    #accArr = partial_accumilate_distribution(SAVE_LOAD_PATH, test_images, test_labels, size)

    full_percent = 0

    if size != None:
        print(f"\n---------------------------------------------------\nFor reselution {size}, the result is as following:\n")

    for i in range(len(accArr)):
        percent = 0 if accArr[i][1] == 0 else 100 - (accArr[i][0] / accArr[i][1]) * 100
        full_percent += percent
        print("Class: {} | Correct: {} | Wrong: {} | percent: {:.2f}".format(str(i).zfill(2), str(accArr[i][1]).rjust(6, ' '), str(accArr[i][0]).rjust(4, ' '), percent))
    #print(f"Pictures in evaluation set: {len(test_images)}, with an average accuracy of: {round(full_percent / len(accArr), 2)}")
