import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import  os
from PIL import  Image
from ml_tool import makePrediction
from Noise_Generators. AddWeather import AddParticels
from Noise_Generators.Perlin_noise import Foggyfy
from Noise_Generators.birghtness import DayAdjustment

def plot_image(i:int, prediction:list, true_label:str, img:str, class_names:list)->None:
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    guessed_rigth = False

    predicted_label = numpy.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
        guessed_rigth = True
    else:
        color = 'red'

    softmaxed = tf.keras.activations.softmax(prediction)

    if guessed_rigth:
        plt.xlabel("CORRECT ({:2.0f}%)\nPrediction: {}".format(100*numpy.max(softmaxed), class_names[predicted_label], color=color))
    else:
        plt.xlabel("WRONG ({:2.0f}%)\nPredictoin: {}\nActual: {}".format(100*numpy.max(softmaxed), class_names[predicted_label], class_names[int(true_label)], color=color))

def plot_value_array(i:int, prediction:list, true_label:str)->None:
    plt.grid(False)
    plt.xticks(range(43))
    plt.yticks([])
    thisplot = plt.bar(range(43), prediction[0], color="#777777")
    plt.ylim([0, 100])
    predicted_label = numpy.argmax(prediction)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')

def predict_and_plot_images(model, class_names:numpy.array, image_dataset:numpy.array, label_datset:numpy.array)->None:
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

def ShowExample(path):
    i = 0
    picArr = []
    """
    for files in os.listdir(path):
        if files.endswith(".ppm"):
            p = path + "/" + files
            img = Image.open(p)
            picArr.append(AddParticels(img.copy()))
            picArr.append(Foggyfy(img))

            print("stuff")
    """    
    
    img = Image.open(path)
    picArr.append(AddParticels(img.copy(), size=4, Opacity=150, frequency=110, LoopJumpX=2, LoopJumpY=2))
    picArr.append(AddParticels(img.copy(), size=4, Opacity=70, frequency=110, LoopJumpX=2, LoopJumpY=2))
    picArr.append(Foggyfy(img.copy()))
    picArr.append(DayAdjustment(img.copy(), 0.5))
    picArr.append(DayAdjustment(img.copy(), 1.9))

    fig=plt.figure(figsize=(4, 12))
    columns = 1
    rows = len(picArr)
    for i in range(1, columns*rows +1):
        print(i)
        img = picArr[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    plt.show()


"""
    index = 0
    for i in range(num_rows):
        for j in range(num_rows):
            axs[i, j].imshow(picArr[index])
            index += 1
            """


#ShowExample("FullIJCNN2013/00")
ShowExample("Images/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/00000/00002_00029.ppm")