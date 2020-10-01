import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy


from data import get_data, split_data, display_numpy_image
from extract import get_class_names
from ml_tool import makePrediction
from show import predict_and_plot_images



SAVE_LOAD_PATH = 'saved_models/YEET7.h5'

img_dataset = [] # list of all images in reshaped numpy array
img_labels = [] # labels for all images in correct order
images_per_class = [] # list, where each entry represents the number of ppm images for that classification class
class_names = [] # classification text for labels

plt.show()

class_names = get_class_names()


img_dataset, img_labels, images_per_class = get_data(fixed_size = (32, 32), padded_images = False, smart_resize = True)
# Training and test split, 70 and 30%
train_images, train_labels, test_images, test_labels = split_data(img_dataset, img_labels, images_per_class, training_split=.7, shuffle=True)

def TrainModel():    
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

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    tf.keras.models.save_model(
        model,
        filepath= SAVE_LOAD_PATH,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
TrainModel()

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

    for i in range(len(accArr)):
        percent = 100 - (accArr[i][0] / accArr[i][1]) * 100
        print(f"Sign label: {str(i).zfill(2)}, Correct: {str(accArr[i][1]).zfill(4)}, Wrong: {str(accArr[i][0]).zfill(4)}, percent: {str(percent).zfill(3)}")
    print(f"Pictures in training set: {len(test_images)}")
AccDistribution()

def TestModel():
    # check the create model
    model = tf.keras.models.load_model(SAVE_LOAD_PATH)
    print(model.evaluate(numpy.array(test_images), numpy.array(test_labels)))
    #display_numpy_image(test_images[0])
    #print(test_labels[0:5])
    #print(model.test_on_batch(test_images,test_labels))

    # check 5 examples
    predict_and_plot_images(model, class_names, test_images[0:5], test_labels[0:5])


# NOW ready to train other model or make predictions on Data

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

"""
plt.figure(figsize=(15,15))
for i in range(20):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    plt.xlabel(class_names[train_labels[i]])
plt.show()

plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
model.save('saved_models/YEET')


# Make a single prediction

#print prediction for test_img1_reshaped
"""

"""
from traffic_cnn import *
model = tf.keras.models.load_model('saved_models/cnn_1')

predict_and_plot_images(model, class_names, test_images[0:5], test_labels[0:5])

plt.figure(figsize=(15,15))
# pick some images from test split between 1000 and 1300
for i in range(3,4):
    plt.subplot(10, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    # predict the image
    test_img_reshaped = tf.reshape(test_images[i], (1, 32, 32, 3))
    predictionList = model.predict_step(test_img_reshaped)
    # The CIFAR labels happen to be arrays,
    plt.xlabel('actual = ' + class_names[test_labels[i]] + ' | prediction = ' + class_names[numpy.argmax(predictionList)])
"""
