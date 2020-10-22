import math
import  tensorflow as tf
from tensorflow.keras import datasets, layers, models

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import  get_model_path

def store_model(model:tf.python.keras.engine.sequential.Sequential, path:str)->None:
    """Saves a model in a given path

    Args:
        model (tf.python.keras.engine.sequential.Sequential): The input model to save
        path (str): The path where the model shold be saved
    """
    tf.keras.models.save_model(
        model,
        filepath= path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )

# def train_model(train_images:list, train_labels:list, test_images:list, test_labels:list, SAVE_LOAD_PATH:str, save_model = True):
#     """Trains a model on agiven train and validation set, and saves if the save_model is true, in an input path

#     Args:
#         train_images (list): The images representing the training set
#         train_labels (list): The labesl for the training images
#         test_images (list): The images representing the validation set
#         test_labels (list): The lable for the validation images
#         SAVE_LOAD_PATH (str): the path to save the model in
#         save_model (bool, optional): Decides whether or not to save the model. Defaults to True.
#     """
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#     model.summary()

#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(43))

#     model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['sparse_categorical_accuracy'])

#     for i in range(10):
#         total_examples = train_images.shape[0]
#         subset = math.floor(total_examples * 0.1)
#         start = math.floor(i * total_examples)
#         end = start + subset
#         print(f'total_examples = {total_examples} | subset = {subset} | start = {start} | end = {end}')
#         history = model.fit(train_images[start:end], train_labels[start:end], epochs=10,
#                         validation_data=(test_images, test_labels))

#     if save_model:
#         store_model(model, SAVE_LOAD_PATH)

def flatten_and_dense(model:tf.python.keras.engine.sequential.Sequential, output_layer_size=62):
    """Returns a model flattened and densed to 43 categories of prediction"""
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_layer_size)) # TODO: IMPORTANT, should be input. Represents the amount of classes
    return model

