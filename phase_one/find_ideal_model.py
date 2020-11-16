import numpy as np
from PIL import Image
import tensorflow as tf
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import trange
import keras.backend as K


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os.path
from os import path

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from error_handler import check_if_valid_path, custom_error_check
from general_image_func import get_class_names, display_numpy_image                        # Not an error
from Models.create_model import flatten_and_dense          # Not an error
from global_paths import get_paths, get_satina_model_avg_path, get_satina_model_median_path, get_satina_model_mode_path, get_satina_model_avg_path_norm, get_satina_model_median_path_norm, get_satina_model_mode_path_norm

class return_model(object):
    """
    docstring
    """
    def __init__(self, model_and_resolution:tuple, model_path:str, out_layer_size:int, load_trained_models:bool)->object:
        self.model, self.img_shape = model_and_resolution
        self.model = flatten_and_dense(self.model, output_layer_size=out_layer_size)
        self.output_layer_size = out_layer_size
        self.path = model_path
        self.fit_data = [['epoch', 'validation_loss', 'validation_accuracy']]
        self.epoch = -1

        if load_trained_models:
            check_if_valid_path(self.path, class_name='return_model')
            self.model = tf.keras.models.load_model(self.path)

        self.csv_data = [['Epochs', 'Resolution', 'Class', 'Class_Acuracy', 'Total_in_Class']]

    def set_epoch(self, epoch:int):
        self.epoch = epoch
        

    def run_on_epoch(self, current_epoch:int)->bool:
        try:
            return_bool = True if int(self.epoch) >= int(current_epoch) or int(self.epoch) == -1 else False
        except ValueError:
            custom_error_check(False, f'Could not convert to integer on {self.epoch}')
        
        return return_bool

    def get_size_tuple(self, last_size:int)->tuple:
        try:
            return_tuple = (self.img_shape[0], self.img_shape[1], last_size) 
        except IndexError:
            custom_error_check(False, f'You are trying to access index one of a list of length {len(self.img_shape)}')
        
        return return_tuple

    def get_size(self)->int:
        try:
            return_value = self.img_shape[0]
        except IndexError:
            custom_error_check(False, f'You are trying to access index one of a list of length {len(self.img_shape)}')
        
        return return_value

    def get_csv_name(self, extension=None)->str:
        if extension == None:
            return f"model_{return_model.get_size(self)}"
        else:
            return f"model_{return_model.get_size(self)}_{extension}"
        
    def get_summed_csv_name(self, extension=None)->str:
        if extension == None:
            return f"model_{return_model.get_size(self)}_summed"
        else:
            return f"model_{return_model.get_size(self)}_{extension}_summed"

    def get_csv_path(self, extension=None)->str:
        if extension == None:
            return f"{get_paths('phase_one_csv')}/{return_model.get_csv_name(self)}.csv"
        else:
            return f"{get_paths('phase_one_csv')}/{return_model.get_csv_name(self)}_{extension}.csv"

    def get_summed_csv_path(self, extension=None)->str:
        if extension == None:
            return f"{get_paths('phase_one_csv')}/{return_model.get_summed_csv_name(self)}.csv"
        else:
            return f"{get_paths('phase_one_csv')}/{return_model.get_summed_csv_name(self, extension=extension)}.csv"

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

def get_satina_median_model_norm()->object:
    img_shape = (55, 55, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    return model, img_shape[:2]

def get_satina_mode_model_norm()->object:
    img_shape = (24, 24, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    return model, img_shape[:2]

def get_satina_avg_model_norm()->object:
    img_shape = (45, 45, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=img_shape))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
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
        progress.set_description(f"Reshaping image {i} / {progress}")
        progress.refresh()

        try:
            reshaped_images.append(tf.keras.preprocessing.image.smart_resize(images[i], size))
        except:
            custom_error_check(False, f'Could not reshape image {i}/{done} to size {size}')

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

    initial_learning_rate = 0.001 

    opt = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam'
    )
    
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False
    )
    
    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    # test_images = test_images
    # train_images = train_images
    
    def lr_exp_decay(epoch, lr):
        k = 0.1
        return initial_learning_rate * math.exp(-k*epoch)

    history = model.fit(train_images, train_labels, epochs=epochs,
            validation_data=(val_images, val_labels),
            callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1),earlystop])
    
    validation_loss = history.history['val_loss']
    validation_accuracy = history.history['val_sparse_categorical_accuracy']
    learning_rate = K.eval(model.optimizer.lr) 
    
    print(f"The initial learning rate is: {initial_learning_rate}, and the final learning rate is: {learning_rate}")
    
    return validation_loss, validation_accuracy



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

    validation_loss, validation_accuracy = train_model(model, reshaped_train_images, train_labels, reshaped_test_images, test_labels, epochs)

    print("\n---------------------")
    print("The test for this model is now done")
    print("---------------------\n")
    
    return validation_loss, validation_accuracy

def verify_model_paths(model_paths):
    if model_paths == None:
        return True
    elif len(model_paths) == 3:
        return True
    else:
        return False

def get_len_if_not_none(model_path):
    return 'NONE' if model_path == None else len(model_path) 

def get_satina_gains_model_object_list(shape:int, load_trained_models:bool=False, model_paths=None)->list:
    custom_error_check(verify_model_paths(model_paths), f'The model path length is not correct. It is {get_len_if_not_none(model_paths)}, but it should be 3')
    
    median_path = get_satina_model_median_path() if model_paths==None else model_paths[0]
    avg_path = get_satina_model_avg_path() if model_paths==None else model_paths[1]
    small_path = get_satina_model_mode_path() if model_paths==None else model_paths[2]
    
    satina_model_avg = return_model(get_satina_avg_model(), median_path, shape, load_trained_models)
    satina_model_median = return_model(get_satina_median_model(), avg_path, shape, load_trained_models)
    satina_model_mode = return_model(get_satina_mode_model(), small_path, shape, load_trained_models)

    return [satina_model_median, satina_model_avg, satina_model_mode]

def get_satina_gains_model_norm_object_list(shape:int, load_trained_models:bool=False, model_paths=None)->list:
    custom_error_check(verify_model_paths(model_paths), f'The model path length is not correct. It is {get_len_if_not_none(model_paths)}, but it should be 3')
    
    median_path = get_satina_model_median_path() if model_paths==None else model_paths[0]
    avg_path = get_satina_model_avg_path() if model_paths==None else model_paths[1]
    small_path = get_satina_model_mode_path() if model_paths==None else model_paths[2]
    
    
    satina_model_avg = return_model(get_satina_avg_model_norm(), median_path, shape, load_trained_models)
    satina_model_median = return_model(get_satina_median_model_norm(), avg_path, shape, load_trained_models)
    satina_model_mode = return_model(get_satina_mode_model_norm(), small_path, shape, load_trained_models)

    return [satina_model_median, satina_model_avg, satina_model_mode]