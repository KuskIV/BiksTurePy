import numpy as np
from PIL import Image
import tensorflow as tf
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import trange


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os.path
from os import path

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from general_image_func import get_class_names, display_numpy_image                        # Not an error
from Models.create_model import flatten_and_dense          # Not an error
from global_paths import get_belgium_model_path, get_paths, get_belgium_model_avg_path, get_belgium_model_median_path, get_satina_model_avg_path, get_satina_model_median_path, get_satina_model_mode_path

class return_model(object):
    """
    docstring
    """
    def __init__(self, model_and_resolution:tuple, model_path:str, out_layer_size:int, load_trained_models:bool)->object:
        self.model, self.img_shape = model_and_resolution
        self.model = flatten_and_dense(self.model, output_layer_size=out_layer_size)
        self.output_layer_size = out_layer_size
        self.path = model_path
        self.epoch = -1

        if load_trained_models:
            if path.exists(self.path):
                self.model = tf.keras.models.load_model(self.path)
            else:
                print(f"The path for the model does not exists ({self.path}, and the program will now exit.)")
                sys.exit()

        self.csv_data = [['Epochs', 'Resolution', 'Class', 'Class_Acuracy', 'Total_in_Class']]

    def set_epoch(self, epoch:int):
        self.epoch = epoch

    def run_on_epoch(self, current_epoch:int)->bool:
        return True if int(self.epoch) >= int(current_epoch) or int(self.epoch) == -1 else False

    def get_size_tuple(self, last_size:int)->tuple:
        return (self.img_shape[0], self.img_shape[1], last_size)

    def get_size(self)->int:
        return self.img_shape[0]

    def get_csv_name(self)->str:
        return f"model{return_model.get_size(self)}"

    def get_csv_path(self)->str:
        return f"{get_paths('phase_one_csv')}/{return_model.get_csv_name(self)}.csv"

    def get_summed_csv_path(self)->str:
        return f"{get_paths('phase_one_csv')}/model{return_model.get_size(self)}_summed.csv"

def get_2d_image_shape(shape:tuple)->tuple:
    return shape[0], shape[1]

def get_satina_median_model()->object:
    img_shape = (55, 55, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    return model, img_shape[:2]

def get_satina_mode_model()->object:
    img_shape = (24, 24, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    return model, img_shape[:2]

def get_satina_avg_model()->object:
    img_shape = (45, 45, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    return model, img_shape[:2]

def get_belgium_model()->object:
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
    return den_danske_model, img_shape[:2]

def get_belgium_model_avg()->object:
    img_shape = (131, 131, 3)
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

    return model, img_shape[:2]

def get_belgium_model_median()->object:
    img_shape = (101, 101, 3)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model, img_shape[:2]

def reshape_numpy_array_of_images(images:np.array, size:tuple)->np.array:
    """Reshapes all images contained in a numpy array, to some specefied size

    Args:
        images (numpy.array): images in numpy array form
        size (tuple): tuple with two integers to specefy the new image shape

    Returns:
        numpy.array: numpy array of the re-sized images
    """
    reshaped_images = []

    done = len(images)
    progress = trange(done, desc='Reshape stuff', leave=True)

    for i in progress:
        progress.set_description(F"Reshaping image {i} / {progress}")
        progress.refresh()

        reshaped_images.append(tf.keras.preprocessing.image.smart_resize(images[i], size))

    return np.array(reshaped_images)

def train_model(model:tf.python.keras.engine.sequential.Sequential,
                train_images:np.array,
                train_labels:np.array,
                val_images:np.array,
                val_labels:np.array,
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

    if len(train_images) == 0 or len(train_labels) == 0 or len(val_images) == 0 or len(val_labels) == 0:
        print(f"ERROR: When training, either the train or validation set contains an empty list.")
        print(f"    - train images     : {len(train_images)}\n")
        print(f"    - train lables     : {len(train_labels)}\n")
        print(f"    - validation images: {len(val_images)}\n")
        print(f"    - validation images: {len(val_labels)}\n")
        sys.exit()

    if len(train_labels) != len(train_images) or len(val_images) != len(val_labels):
        print(f"ERROR: the image and label lists are not the same size:")
        print(f"    - train images : {len(train_images)} - {len(train_labels)} : train lables")
        print(f"    - validation images : {len(val_images)} - {len(val_labels)} : validation lables")
        sys.exit()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    # test_images = test_images
    # train_images = train_images

    history = model.fit(train_images, train_labels, epochs=epochs,
            validation_data=(val_images, val_labels))


def train_and_eval_models_for_size(
        size:tuple,
        model:tf.python.keras.engine.sequential.Sequential,
        train_images:np.array,
        train_labels:np.array,
        test_images:np.array,
        test_labels:np.array,
        epochs=10,
        )->None:
    """Trains and evaluates models based on the size of images used

    Args:
        models (list): list of models
        size (tuple): size of the models
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
    print("\n---------------------")
    print("The model will now train with the following image size:")
    print(size)
    print("---------------------\n")

    train_model(model, reshaped_train_images, train_labels, reshaped_test_images, test_labels, epochs)

    # evaluate each model
    # print("Evaluation for model")

    # print(model.evaluate(reshaped_test_images, test_labels))

def get_satina_gains_model_object_list(shape:int, load_trained_models:bool=False)->list:
    satina_model_avg = return_model(get_satina_avg_model(), get_satina_model_avg_path(), shape, load_trained_models)
    satina_model_median = return_model(get_satina_median_model(), get_satina_model_median_path(), shape, load_trained_models)
    satina_model_mode = return_model(get_satina_mode_model(), get_satina_model_mode_path(), shape, load_trained_models)

    return [satina_model_avg, satina_model_median, satina_model_mode]

def get_belgian_model_object_list(shape:int, load_trained_models=False)->list:
    belgium_model_avg = return_model(get_belgium_model_avg(), get_belgium_model_avg_path(), shape, load_trained_models)
    belgium_model_median = return_model(get_belgium_model_median(), get_belgium_model_median_path(), shape, load_trained_models)
    belgium_model = return_model(get_belgium_model(), get_belgium_model_path(), shape, load_trained_models)

    return [belgium_model]
    #return [belgium_model_avg, belgium_model_median, belgium_model]

def load_best_model_and_image_size(model_path:str)->tuple:
    model = tf.keras.models.load_model(model_path)
    return model, model.input_shape[1:3]


def get_best_phase_one_model(shape:int)->object:
    model_path = get_paths('ex_one_ideal')
    return return_model(load_best_model_and_image_size(model_path), model_path, shape, True)
