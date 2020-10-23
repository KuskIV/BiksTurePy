import numpy as np
from PIL import Image
import tensorflow as tf
import math
from matplotlib import pyplot as plt


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import get_class_names, display_numpy_image                        # Not an error
from Models.create_model import flatten_and_dense          # Not an error
from global_paths import get_small_model_path, get_medium_model_path, get_large_model_path, get_belgium_model_path


class return_model(object):
    """
    docstring
    """
    def __init__(self, get_model, path, out_layer_size):
        self.model, self.img_shape = get_model()
        self.model = flatten_and_dense(self.model, output_layer_size=out_layer_size)
        self.path = path

    def get_size_tuple(self, last_size:int):
        return (self.img_shape[0], self.img_shape[1], last_size)

    def get_size(self):
        return self.img_shape[0]




# def default_model()->tf.python.keras.engine.sequential.Sequential:
#     """default model also known as small model since it uses image of the size 32x32.

#     Returns:
#         tf.python.keras.engine.sequential.Sequential: A cnn trained on 32x32 images
#     """
#     img_shape=(32, 32, 3)
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     return model

# def medium_model()->tf.python.keras.engine.sequential.Sequential:
#     """medium model since it uses image of the size 128x128.

#     Returns:
#         tf.python.keras.engine.sequential.Sequential: A cnn trained on 128x128 images
#     """
#     img_shape=(128, 128, 3)
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (15, 15), activation='relu', input_shape=img_shape))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     return model

# def large_model()->tf.python.keras.engine.sequential.Sequential:
#     """large model since it uses image of the size 200x200.

#     Returns:
#         tf.python.keras.engine.sequential.Sequential: A cnn trained on 200x200 images
#     """
#     img_shape=(200, 200, 3)
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (21, 21), activation='relu', input_shape=img_shape))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     return model

def get_2d_image_shape(shape:tuple)->tuple:
    return shape[0], shape[1]

def get_belgium_model():
    img_shape = (82,82,3)
    den_danske_model = models.Sequential()
    den_danske_model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=img_shape))
    den_danske_model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='valid'))
    den_danske_model.add(layers.MaxPool2D((2,2)))
    den_danske_model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    den_danske_model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='valid'))
    den_danske_model.add(layers.MaxPool2D((2,2)))
    den_danske_model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    den_danske_model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='valid'))
    den_danske_model.add(layers.MaxPool2D((2,2)))
    den_danske_model.add(layers.Conv2D(64, (4, 4), activation='relu'))
    return den_danske_model, (get_2d_image_shape(img_shape))


def get_default_model():
    img_shape=(32, 32, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model, (get_2d_image_shape(img_shape))

def get_medium_model():
    img_shape=(128, 128, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model, (get_2d_image_shape(img_shape))

def get_large_model():
    img_shape=(200, 200, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (5, 5), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model, (get_2d_image_shape(img_shape))
    



# def large_model():
#     img_shape=(200, 200, 3)
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_shape))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     return model


def reshape_numpy_array_of_images(images:np.array, size:tuple)->np.array:
    """Reshapes all images contained in a numpy array, to some specefied size

    Args:
        images (numpy.array): images in numpy array form
        size (tuple): tuple with two integers to specefy the new image shape

    Returns:
        numpy.array: numpy array of the re-sized images
    """
    reshaped_images = []

    for image in images:
        reshaped_images.append(tf.keras.preprocessing.image.smart_resize(image, size))
    
    return np.array(reshaped_images)

def train_model(model:tf.python.keras.engine.sequential.Sequential, 
                train_images:np.array,
                train_labels:np.array, 
                test_images:np.array, 
                test_labels:np.array,
                epochs:int)->None:
    """Method for using the data set on some input model

    Args:
        model (tf.python.keras.engine.sequential.Sequential): A previoslt created model,
         which is the trained using the inputette dataset

        train_images (numpy.array): training images
        train_labels (numpy.array): labels for the training images
        test_images (numpy.array): test images
        test_labels (numpy.array): labels for the test images
    """
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
    
    test_images = test_images
    train_images = train_images #TODO ask toi about this
    
    history = model.fit(train_images, train_labels, epochs=epochs,
            validation_data=(test_images, test_labels))


def train_and_eval_models_for_size(#TODO pls help
        size:int,
        model:tf.python.keras.engine.sequential.Sequential,
        model_id:int,
        train_images:np.array,
        train_labels:np.array,
        test_images:np.array,
        test_labels:np.array,
        epochs=10,
        save_model=True)->None:
    """Trains and evaluates models based on the size of images used

    Args:
        models (list): list of models
        size (list): size of the models
        model (tf.python.keras.engine.sequential.Sequential): input model
        model_id (int): index
        train_images (numpy.array): training images
        train_labels (numpy.array): training labels
        test_images (numpy.array): test images
        test_labels (numpy.array): test labels
        save_model (bool, optional): wheter it should save ot not. Defaults to True.
    """
    # reshape training and test images
    reshaped_train_images = reshape_numpy_array_of_images(train_images, size)
    reshaped_test_images = reshape_numpy_array_of_images(test_images, size)


    #print(type(train_labels), " ", type(train_labels[0]), " ", type(test_labels), " ", type(test_labels[0]))
    
    # train model
    print("image size")
    print(size)
    train_model(model, reshaped_train_images, train_labels, reshaped_test_images, test_labels, epochs)

    # evaluate each model
    print("Evaluation for model")

    print(reshaped_test_images.shape, "  ", reshaped_train_images[0].shape)
    #print(model.evaluate(reshaped_test_images, test_labels))


def get_model_object_list(shape:int):
    large_model = return_model(get_large_model, get_large_model_path(), shape)
    medium_model = return_model(get_medium_model, get_medium_model_path(), shape)
    small_model = return_model(get_default_model, get_small_model_path(), shape)
    belgium_model = return_model(get_belgium_model, get_belgium_model_path(), shape)

    return [belgium_model, small_model]

# def get_processed_models(input_layer_size=62):
#     large_model, large_size, large_path = get_large_model()
#     medium_model, medium_size, medium_path = get_medium_model()
#     default_model, default_size, default_path = get_default_model()
#     belgium_model, belgium_size, belgium_path = get_belgium_model()

#     return [(flatten_and_dense(large_model, input_layer_size=input_layer_size), large_size), 
#             (flatten_and_dense(medium_model, input_layer_size=input_layer_size), medium_size), 
#             (flatten_and_dense(default_model, input_layer_size=input_layer_size), default_size),
#             (flatten_and_dense(belgium_model, input_layer_size=input_layer_size), belgium_size)]
