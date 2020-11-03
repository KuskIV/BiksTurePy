import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from Dataset.load_h5 import h5_object
from  phase_one.find_ideal_model import get_belgian_model_object_list
import global_paths
from general_image_func import auto_reshape_images
from phase_one.find_ideal_model import reshape_numpy_array_of_images
print(tf.__version__)

# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
h5_train = h5_object(global_paths.get_paths('h5_train'), training_split=0.8)
train_images, train_labels, test_images, test_labels = h5_train.shuffle_and_lazyload(0, 2)


train_images = reshape_numpy_array_of_images(train_images,(82,82))
test_images =reshape_numpy_array_of_images(test_images,(82,82))

train_images.shape

len(train_labels)

train_labels

test_images.shape

# def to_rgb1(im):
#     # I think this will be slow
#     w, h = im.shape
#     ret = np.empty((w, h, 3), dtype=np.uint8)
#     ret[:, :, 0] = im
#     ret[:, :, 1] = im
#     ret[:, :, 2] = im
#     return ret
# # lst = []
# # for i in range(len(train_images)):
# #   lst.append(to_rgb1(train_images[i]))
# # train_images = np.array(lst)
# # lst = []
# # for i in range(len(test_images)):
# #   lst.append(to_rgb1(test_images[i]))
# # test_images = lst
# # test_images = np.array(lst)
# # lst = []

# train_images = train_images / 255.0
# test_images = test_images / 255.0

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(82, 82,3)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(107)
# ])
model_object = get_belgian_model_object_list(107)[0]

model = model_object.model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
model.fit(train_images, train_labels, epochs=100,validation_data=(test_images,test_labels))
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

img = test_images[1]

print(img.shape)

img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])