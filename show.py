import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

from ml_tool import makePrediction




def plot_image(i, prediction, true_label, img, class_names):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    predicted_label = numpy.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    softmaxed = tf.keras.activations.softmax(prediction)


    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*numpy.max(softmaxed),
                                         class_names[int(true_label)],
                                         color=color))

def plot_value_array(i, prediction, true_label):
    plt.grid(False)
    plt.xticks(range(43))
    plt.yticks([])
    thisplot = plt.bar(range(43), prediction[0], color="#777777")
    plt.ylim([0, 100])
    predicted_label = numpy.argmax(prediction)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')

def predict_and_plot_images(model, class_names, image_dataset, label_datset):
    """Insert a model, list of class_names, list of numpy images, list of numpy labels"""
    num_rows = len(image_dataset)
    num_cols = 1
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*9*num_cols, 2*num_rows))
    for i in range(num_images):
        label = label_datset[i]
        image = image_dataset[i]
        prediction = makePrediction(model, image)


        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, prediction, label, image, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, prediction, label)
    plt.tight_layout()
    plt.show()
