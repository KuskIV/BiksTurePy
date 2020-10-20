import tensorflow as tf
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import math
import  os
from PIL import  Image

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import make_prediction

def plot_image(i:int, prediction:list, true_label:str, img:str, class_names:list)->None:
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    guessed_rigth = False

    predicted_label = numpy.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
        guessed_rigth = True
    else:
        color = 'red'

    softmaxed = tf.keras.activations.softmax(prediction)

    if guessed_rigth:
        plt.xlabel("CORRECT ({:2.0f}%)\nPrediction: {}".format(100*numpy.max(softmaxed), class_names[predicted_label], color=color))
    else:
        plt.xlabel("WRONG ({:2.0f}%)\nPredictoin: {}\nActual: {}".format(100*numpy.max(softmaxed), class_names[predicted_label], class_names[int(true_label)], color=color))

def plot_value_array(i:int, prediction:list, true_label:str)->None:
    plt.grid(False)
    plt.xticks(range(43))
    plt.yticks([])
    thisplot = plt.bar(range(43), prediction[0], color="#777777")
    plt.ylim([0, 100])
    predicted_label = numpy.argmax(prediction)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')

def predict_and_plot_images(model, class_names:numpy.array, image_dataset:numpy.array, label_datset:numpy.array)->None:
    """Insert a model, list of class_names, list of numpy images, list of numpy labels"""
    num_rows = len(image_dataset)
    num_cols = 1
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*9*num_cols, 2*num_rows))
    for i in range(num_images):
        label = label_datset[i]
        image = image_dataset[i]
        prediction = make_prediction(model, image)

        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, prediction, label, image, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, prediction, label)
    plt.tight_layout()
    plt.show()
