import numpy
import tensorflow as tf
import math

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from data import get_data, split_data, display_numpy_image
from extract import get_class_names
from ml_tool import *
#from show import predict_and_plot_images

SAVE_LOAD_PATH = 'saved_models/YEET8.h5'
STORE_FOLDER_NAME = 'saved_models/'


def TrainModel(save_model = True):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(43))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    for i in range(10):
        total_examples = train_images.shape[0]
        subset = math.floor(total_examples * 0.1)
        start = math.floor(i * total_examples)
        end = start + subset
        print(f'total_examples = {total_examples} | subset = {subset} | start = {start} | end = {end}')
        history = model.fit(train_images[start:end], train_labels[start:end], epochs=10,
                        validation_data=(test_images, test_labels))

    if save_model:
        tf.keras.models.save_model(
            model,
            filepath= SAVE_LOAD_PATH,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )

def store_model(model, filename):
    tf.keras.models.save_model(
        model,
        filepath= STORE_FOLDER_NAME+filename,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

def AccDistribution():
    model = tf.keras.models.load_model(SAVE_LOAD_PATH)
    model.summary()

    accArr = numpy.zeros((43, 2))

    for i in range(len(test_images)):
        prediction = makePrediction(model, test_images[i])
        softmaxed = tf.keras.activations.softmax(prediction)
        if test_labels[i] == numpy.argmax(softmaxed):
            accArr[int(test_labels[i])][1] = accArr[int(test_labels[i])][1] + 1
        else:
            accArr[int(test_labels[i])][0] = accArr[int(test_labels[i])][0] + 1
    full_percent = 0
    for i in range(len(accArr)):
        percent = 100 - (accArr[i][0] / accArr[i][1]) * 100
        full_percent += percent
        print("Class: {} | Correct: {} | Wrong: {} | percent: {:.2f}".format(str(i).zfill(2), str(accArr[i][1]).rjust(6, ' '), str(accArr[i][0]).rjust(4, ' '), percent))
    #print(f"Pictures in evaluation set: {len(test_images)}, with an average accuracy of: {round(full_percent / len(accArr), 2)}")

def TestModel():
    # check the create model
    model = tf.keras.models.load_model(SAVE_LOAD_PATH)
    print(model.evaluate(numpy.array(test_images), numpy.array(test_labels)))
    #display_numpy_image(test_images[0])
    #print(test_labels[0:5])
    #print(model.test_on_batch(test_images,test_labels))

    # check 5 first examples in test set
    predict_and_plot_images(model, class_names, test_images[0:5], test_labels[0:5])


# NOW ready to train other model or make predictions on Data

def default_model():
    img_shape=(32, 32, 3)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model

def medium_model():
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

def large_model():
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

def flatten_and_dense(model):
    """Returns a model flattened and densed to 43 categories of prediction"""
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(43))
    return model


def reshape_numpy_array_of_images(images, size):
    reshaped_images = []
    for image in images:
        reshaped_images.append(tf.keras.preprocessing.image.smart_resize(image, size))
    return numpy.array(reshaped_images)

def train_model(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
            validation_data=(test_images, test_labels))


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

def train_and_eval_models_for_size(size, model, model_id, train_images, train_labels, test_images, test_labels, save_model=True):
    if size != (32, 32):
        # reshape training and test images
        reshaped_train_images = reshape_numpy_array_of_images(train_images, size)
        reshaped_test_images = reshape_numpy_array_of_images(test_images, size)
    else:
        reshaped_train_images = train_images #set to default
        reshaped_test_images = test_images #set to default

    # train model
    print("image size")
    print(size)
    train_model(model, reshaped_train_images, train_labels, reshaped_test_images, test_labels)

    # evaluate each model
    print("Evaluation for model")
    print(model.evaluate(reshaped_test_images, test_labels))

    # stor each model
    if save_model:
        #store model in saved_models with name as img_shape X model design
        filename = 'adj'+str(size[0])
        model_id = models.index(model)
        if model_id == 0:
            filename += "default"
        elif model_id == 1:
            filename += "medium"
        else:
            filename += "large"
        store_model(model, filename)

# Below is executed when file is executed directly as main
if __name__ == "__main__":
    img_dataset = [] # list of all images in reshaped numpy array
    img_labels = [] # labels for all images in correct order
    images_per_class = [] # list, where each entry represents the number of ppm images for that classification class
    class_names = [] # classification text for labels

    class_names = get_class_names()

    image_sizes = [(32, 32), (128, 128), (200, 200)]

    img_dataset, img_labels, images_per_class = get_data(fixed_size = (0, 0), padded_images = False, smart_resize = True, normalize=False)


    # Training and test split, 70 and 30%
    train_images, train_labels, test_images, test_labels = split_data(img_dataset, img_labels, images_per_class, training_split=.7, shuffle=True)

    analyze_data(train_images, train_labels, images_per_class)


    """
    # generate models
    models = [flatten_and_dense(default_model()), flatten_and_dense(medium_model()), flatten_and_dense(large_model())]

    # zip together with its size
    model_and_size = list(zip(models, image_sizes))

    # train models
    for i in range(len(model_and_size)):
        train_and_eval_models_for_size(model_and_size[i][1], model_and_size[i][0], i, train_images, train_labels, test_images, test_labels)
    """
