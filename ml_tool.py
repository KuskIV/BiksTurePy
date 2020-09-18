import tensorflow as tf

def makePrediction(model, image):
    img_reshaped = tf.reshape(image, (1, 32, 32, 3))
    return model.predict_step(img_reshaped)
