import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy

from data import get_data, split_data
from extract import get_class_names
from ml_tool import makePrediction
from show import predict_and_plot_images

img_dataset = [] # list of all images in reshaped numpy array
img_labels = [] # labels for all images in correct order
images_per_class = [] # list, where each entry represents the number of ppm images for that classification class
class_names = [] # classification text for labels

class_names = get_class_names()

img_dataset, img_labels, images_per_class = get_data(fixed_size = (32, 32), padded_images = False, smart_resize = True)

# Training and test split, 70 and 30%
train_images, train_labels, test_images, test_labels = split_data(img_dataset, img_labels, training_split=.7, shuffle=True)

# NOW ready to train other model or make predictions on Data

# check the create model
model = tf.keras.models.load_model('saved_models/YEET2')

# check 5 examples
predict_and_plot_images(model, class_names, test_images[0:5], test_labels[0:5])

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
model.save('saved_models/YEET')
tf.keras.models.save_model(
    model,
    filepath= 'saved_models/YEET2',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

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
