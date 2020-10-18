import numpy as np
import tensorflow as tf
from PIL import Image

def make_prediction(model:tf.python.keras.engine.sequential.Sequential, image:Image.Image)->list:
    """Based on one image, a model makes a prediction to what it is

    Args:
        model (tf.python.keras.engine.sequential.Sequential): The model to use
        image (Image.Image): The imaage to predict on

    Returns:
        list: The probability of the given image beign each class. Softmax not yet applied
    """
    img_reshaped = tf.reshape(image, (1, 32, 32, 3))
    return model.predict_step(img_reshaped)

def accumilate_distribution(SAVE_LOAD_PATH:str, test_images:list, test_labels:list)->None:
    """Given a set of images and lables, a prediction is made for each image

    Args:
        SAVE_LOAD_PATH (str): The path to load the model from
        test_images (list): A set of images
        test_labels (list): A set of lables
    """

    model = tf.keras.models.load_model(SAVE_LOAD_PATH)
    model.summary()

    accArr = np.zeros((43, 2))

    for i in range(len(test_images)):
        prediction = make_prediction(model, test_images[i])
        softmaxed = tf.keras.activations.softmax(prediction)
        if test_labels[i] == np.argmax(softmaxed):
            accArr[int(test_labels[i])][1] = accArr[int(test_labels[i])][1] + 1
        else:
            accArr[int(test_labels[i])][0] = accArr[int(test_labels[i])][0] + 1

    full_percent = 0

    for i in range(len(accArr)):
        percent = 100 - (accArr[i][0] / accArr[i][1]) * 100
        full_percent += percent
        print("Class: {} | Correct: {} | Wrong: {} | percent: {:.2f}".format(str(i).zfill(2), str(accArr[i][1]).rjust(6, ' '), str(accArr[i][0]).rjust(4, ' '), percent))
    #print(f"Pictures in evaluation set: {len(test_images)}, with an average accuracy of: {round(full_percent / len(accArr), 2)}")
