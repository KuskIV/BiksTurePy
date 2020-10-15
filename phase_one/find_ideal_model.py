import numpy
import tensorflow as tf
import math

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import get_class_names                          # Not an error
from Models.create_model import flatten_and_dense          # Not an error

def default_model()->tf.python.keras.engine.sequential.Sequential:
    img_shape=(32, 32, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model

def medium_model()->tf.python.keras.engine.sequential.Sequential:
    img_shape=(128, 128, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (15, 15), activation='relu', input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model

def large_model()->tf.python.keras.engine.sequential.Sequential:
    img_shape=(200, 200, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (21, 21), activation='relu', input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model

def reshape_numpy_array_of_images(images, size):
    reshaped_images = []
    #print(type(images[0]), " ", type(images[0][0]), " ", type(images[0][0][0]), " ", type(images[0][0][0][0]))
    for image in images:
        reshaped_images.append(tf.keras.preprocessing.image.smart_resize(image, size))
    return numpy.array(reshaped_images)

def train_model(model, train_images, train_labels, test_images, test_labels): #TODO: why is this outcommented, probably ask milad
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
            validation_data=(test_images, test_labels))
    print (train_images[0].shape, " ---")


            #print(f'batch number: {i}')

           # for i in range(epoch_size):
            #    train_data = model.train_on_batch(train_images[start:end], test_labels[start:end])

            #[print(element) for element in train_data]

            #if i % batches == 0:
            #    print("897 have been trained")

           # print(f'total_examples = {total_examples} | subset = {subset} | start = {start} | end = {end}')
           # history = model.fit(train_images[start:end], train_labels[start:end], epochs=10,
            #                validation_data=(test_images, test_labels))



    #history = model.fit(train_images, train_labels, epochs=10,
    #                    validation_data=(test_images, test_labels))

def train_and_eval_models_for_size(models, size, model, model_id, train_images, train_labels, test_images, test_labels, save_model=True):
        # reshape training and test images
    reshaped_train_images = reshape_numpy_array_of_images(train_images, size)
    reshaped_test_images = reshape_numpy_array_of_images(test_images, size)
    #print(type(train_labels), " ", type(train_labels[0]), " ", type(test_labels), " ", type(test_labels[0]))
    # train model
    print("image size")
    print(size)
    train_model(model, reshaped_train_images, train_labels, reshaped_test_images, test_labels)

    # evaluate each model
    print("Evaluation for model")

    print(reshaped_test_images.shape, "  ", reshaped_train_images[0].shape)
    #print(model.evaluate(reshaped_test_images, test_labels))

    


def get_processed_models():
    return [flatten_and_dense(large_model()), flatten_and_dense(medium_model()), flatten_and_dense(default_model())]